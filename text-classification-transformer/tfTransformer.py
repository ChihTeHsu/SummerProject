import numpy as np
import tensorflow as tf

# %%
# 現在讓我們看看 scaled dot product attention 在 TensorFlow 裡是怎麼被實作的
def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead) 
    but it must be broadcastable for addition.
    
    Args:
        q: query shape == (..., seq_len_q, depth)
        k: key shape == (..., seq_len_k, depth)
        v: value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable 
            to (..., seq_len_q, seq_len_k). Defaults to None.
    
    Returns:
        output, attention_weights
    """
    # 將 `q`、 `k` 除了 batch 維度外做內積，最後變成 (..., seq_len_q, seq_len_k) 的 tensor 
    # 也就是句子 q 裡頭每個子詞對句子 k 裡頭的每個子詞延（word embedding）空間做內積
    # 下面程式碼等同 tf.matmul(q, tf.transpose(k,perm=[0,2,1])))
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    dk = tf.cast(tf.shape(k)[-1], tf.float32)  # 取得 seq_k 的詞嵌入空間維度 (depth)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)  # scale by sqrt(dk)
    # 除以 scaling factor 的目的是為了讓內積出來的值不會因為 Q 以及 K 的詞嵌入空間維度太大而跟著太大
    # 因為太大的值丟入 softmax 函式有可能使得其梯度變得極小，導致訓練結果不理想
    
    # 將遮罩「加」到被丟入 softmax 前的 logits
    # 效果等同於序列 q 中的某個子詞 sub_q 完全沒放注意力在這些被遮罩蓋住的子詞 sub_k 之上
    if mask is not None:
        scaled_attention_logits += (mask * -1e9) #讓被加上極大負值的位置變得無關緊要，在經過 softmax 以後的值趨近於 0
    
    # 取 softmax 是為了得到總和為 1 的比例之後對 `v` 做加權平均
    # 最後一個維度 axis=-1 代表某個序列 q 裡的某個子詞與序列 k 的每個子詞的匹配程度
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    
    # 以注意權重對 v 做加權平均（weighted average）
    output = tf.matmul(attention_weights, v)  # 最後變回 (..., seq_len_q, depth_v) 的 tensor
    
    # output 代表注意力機制的結果，物理含義是裡頭的每個子詞都獲得了一個包含自己以及周遭子詞語義資訊的新 representation
    # 解讀為 q 在關注 k 並從 v 得到上下文訊息後的所獲得的新 representation 故注意函式的輸出 output張量維度會跟 q 張量相同
    # attention_weights 代表句子 q 裡頭每個子詞對句子 k 裡頭的每個子詞的注意權重
    # 序列 q 為 Decoder 的輸出序列而序列 k 為 Encoder 的輸出序列時，我們就變成在計算一般 Seq2Seq 模型中的注意力機制
    return output, attention_weights

# 設定 padding mask 讓 Transformer 用來識別序列實際的內容到哪裡。
# 此 padding mask 遮罩負責的就是將序列中被補 0 的地方蓋住，讓 Transformer 可以避免「關注」到這些位置
def create_padding_mask(seq):
    # 我們的輸入代表著索引序列的詞嵌入張量，而裡頭應該是有經過補齊每個batch長度的 <pad> token
    # 我們希望序列裡頭的每個子詞 sub_k 都是實際存在的中文字或英文詞彙，所以把索引序列中為 0 的位置設為 1 (蓋住)
    mask = tf.cast(tf.equal(seq, 0), tf.float32)
    # 在 padding mask 中間加入兩個新維度為了之後可以做 broadcasting：
    # 一個是用來遮住同個句子但是不同 head 的注意權重，一個則是用來 broadcast 到 2 維注意權重的
    return mask[:, tf.newaxis, tf.newaxis, :]

# 設定一種遮罩 look ahead mask 遮住 Decoder 將在未來生成的子詞不讓之前的子詞關注
# look ahead 遮罩就是一個 2 維矩陣，其兩個維度都跟輸入的詞嵌入張量的倒數第 2 個維度（序列長度）一樣
def create_look_ahead_mask(size):
    # look_ahead_mask 遮罩為一個右上角的三角形
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    # 此遮罩是用來確保 Decoder 在進行自注意力機制時輸出序列裡頭的每個子詞只會關注到自己之前（左邊）的字詞
    # 不會不小心關注到未來（右邊）理論上還沒被 Decoder 生成的子詞。所以 Decoder 在生成第 1 個子詞時只看自己；
    # 在生成第 2 個子詞時關注前 1 個子詞與自己；在生成第 3 個子詞時關注前兩個已經生成的子詞以及自己，以此類推
    return mask  # (seq_len, seq_len)

# %%
# 實作一個執行多頭注意力機制的 keras layer
# 在初始的時候指定輸出維度 `d_model` & `num_heads，
# 在呼叫的時候輸入 `v`, `k`, `q` 以及 `mask`
# 輸出跟 scaled_dot_product_attention 函式一樣有兩個：
# output.shape            == (batch_size, seq_len_q, d_model)
# attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
class MultiHeadAttention(tf.keras.layers.Layer):
    # 在初始的時候建立一些必要參數
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads # 指定要將 `d_model` 拆成幾個 heads
        self.d_model = d_model # 在 split_heads 之前的基底維度
        
        assert d_model % self.num_heads == 0  # 我們要確保維度 `d_model` 可以被平分成 `num_heads` 個 `depth` 維度
        
        self.depth = d_model // self.num_heads  # 分成多頭以後每個 head 裡子詞的新的 repr. 向量的維度
        
        # 將詞彙轉換到一個 d_model 維的詞嵌入（word embedding）空間
        self.wq = tf.keras.layers.Dense(d_model)  # 分別給 q, k, v 的 3 個線性轉換 
        self.wk = tf.keras.layers.Dense(d_model)  # 注意我們並沒有指定 activation func
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)  # 多 heads 串接後通過的線性轉換
    
    # 實作Multi-head attention 讓它可以同時關注不同位置的子詞在不同子空間下的 representation
    # 要實現 multi-head attention 就是把原先 d_model 維度的詞嵌入向量「折」成 num_heads 個 depth 維向量
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        
        # 將最後一個 d_model 維度分成 num_heads 個 depth 維度。
        # 原先的 3 維詞嵌入張量 x 已經被轉換成一個 4 維張量了，且最後一個維度 shape[-1] 被拆成兩半
        # (batch_size, seq_len, num_heads, depth)
        x = tf.reshape(x, shape=(batch_size, -1, self.num_heads, self.depth))
        
        # 將 head 的維度拉前使得最後兩個維度為子詞以及其對應的 depth 向量 (batch_size, num_heads, seq_len, depth)
        # 所以每個 head 的 2 維矩陣事實上仍然代表原來的序列，只是裡頭子詞的 repr. 維度降低了
        # 原來句子裡頭的子詞現在不只會有一個 d_model 的 repr.，而是會有 num_heads 個 depth 維度的 representation
        return tf.transpose(x, perm=[0, 2, 1, 3])
  
    # multi-head attention 的實際執行流程，注意參數順序（這邊跟論文以及 TensorFlow 官方教學一致）
    def call(self, v, k, q, mask):
        # q.shape: (batch_size, seq_len, d_model)
        batch_size = tf.shape(q)[0]
        
        # 將輸入的 q, k, v 都各自做一次線性轉換到 `d_model` 維空間
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)
        
        # 前面看過的，將最後一個 `d_model` 維度分成 `num_heads` 個 `depth` 維度
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)
        
        # 利用 broadcasting 讓每個句子的每個 head 的 qi, ki, vi 都各自進行注意力機制
        # 輸出會多一個 head 維度
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # 跟我們在 `split_heads` 函式做的事情剛好相反，先做 transpose 再做 reshape
        # 將 `num_heads` 個 `depth` 維度串接回原來的 `d_model` 維度
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(scaled_attention, shape=(batch_size, -1, self.d_model)) 
        # (batch_size, seq_len_q, d_model)
        
        # 通過最後一個線性轉換
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        
        # 輸出張量 output 裡頭每個句子的每個字詞的 repr. 維度 d_model 雖然跟函式的輸入張量相同，
        # 但實際上已經是從同個序列中不同位置且不同空間中的 repr. 取得語義資訊的結果
        return output, attention_weights

# Encoder layer 跟 Decoder layer 裡頭都各自有一個 Position-wise Feed-Forward Networks
# 建立 Transformer 裡 Encoder / Decoder layer 都有使用到的這個 Feed Forward 元件
def point_wise_feed_forward_network(d_model, dff):
    # 此 FFN 對輸入做兩個線性轉換，中間加了一個 ReLU activation func
    # 一般會讓中間層的維度 dff 大於 d_model 的維度
    # Input -> layer1 -> layer2 = output
    # [seq_len, d_model] -> [d_model, dff] -> [diff, d_model] = [seq_len, d_model]
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # 這步後得到 (batch_size, seq_len, dff) 維度 tensor
        tf.keras.layers.Dense(d_model)  # 這步後得到 (batch_size, seq_len, d_model) 維度 tensor
        ])

# %%
# 現在讓我們看看 Encoder layer 的實作：
# Encoder 裡頭會有 N 個 EncoderLayers，而每個 EncoderLayer 裡又有兩個 sub-layers: MHA & FFN
# 我們可以為 FFN 設置不同的 dff 值，也能設定不同的 num_heads 來改變 MHA 內部每個 head 裡頭的維度
class EncoderLayer(tf.keras.layers.Layer):
    # Transformer 論文內預設 dropout rate 為 0.1
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        # layer norm 很常在 RNN-based 的模型被使用。一個 sub-layer 一個 layer norm
        # 針對最後一維 d_model 做 normalization，使其平均與標準差分別靠近 0 和 1 之後輸出
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # 一樣，一個 sub-layer 一個 dropout layer
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
    
    # 需要丟入 `training` 參數是因為 dropout 在訓練以及測試的行為有所不同
    def call(self, x, training, mask):
        # 除了 `attn`，其他張量的 shape 皆為 (batch_size, input_seq_len, d_model)
        # attn.shape == (batch_size, num_heads, input_seq_len, input_seq_len)
        
        # sub-layer 1: MHA
        # Encoder 利用注意機制關注自己當前的序列，因此 v, k, q 全部都是自己
        # 另外別忘了我們還需要 padding mask 來遮住輸入序列中的 <pad> token
        attn_output, attn = self.mha(x, x, x, mask)  
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output) #殘差連結（residual connection）幫助減緩梯度消失的問題
        
        # sub-layer 2: FFN
        ffn_output = self.ffn(out1) 
        ffn_output = self.dropout2(ffn_output, training=training)  # 記得 training
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2

# 接著讓我們看看 DecoderLayer 的實作，
# 一個 Decoder layer 裡頭有 3 個 sub-layers: : 自注意的 MHA, 關注 Encoder 輸出的 MHA & FFN
# 1. 自身的 Masked MHA 1 用來關注輸出序列，查詢 Q、鍵值 K 以及值 V 都是自己
# 2. MHA1 處理完的輸出序列會成為這層的 Q，而 K 與 V 則使用 Encoder 的輸出序列
class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(DecoderLayer, self).__init__()
        
        # 3 個 sub-layers 的主角們
        self.mha1 = MultiHeadAttention(d_model, num_heads)
        self.mha2 = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        # 定義每個 sub-layer 用的 LayerNorm
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        
        # 定義每個 sub-layer 用的 Dropout
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)
    
    def call(self, x, enc_output, training, combined_mask, inp_padding_mask):
        # 所有 sub-layers 的主要輸出皆為 (batch_size, target_seq_len, d_model)
        # enc_output 為 Encoder 輸出序列，shape 為 (batch_size, input_seq_len, d_model)
        # attn_weights_block_1 則為 (batch_size, num_heads, target_seq_len, target_seq_len)
        # attn_weights_block_2 則為 (batch_size, num_heads, target_seq_len, input_seq_len)
        
        # sub-layer 1: Decoder layer 自己對輸出序列做注意力。
        # 因為是關注自己，multi-head attention 的參數 v、k 以及 q 都是 x
        # 我們同時需要 look ahead mask 以及輸出序列的 padding mask 
        # 來避免前面已生成的子詞關注到未來的子詞以及 <pad>
        # 因此 Decoder layer 預期的遮罩是兩者結合的 combined_mask !!!
        attn1, attn_weights_block1 = self.mha1(x, x, x, combined_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x)
        
        # sub-layer 2: Decoder layer 關注 Encoder 的最後輸出輸出序列
        # 因此，multi-head attention 的參數 v、k 為 enc_output，q 則為 MHA 1 sub-layer 的結果 out1
        # 概念是讓一個 Decoder layer 在生成新的中文子詞時先參考先前已經產生的中文字，
        # 並為當下要生成的子詞產生一個包含前文語義的 repr. 。接著將此 repr. 拿去跟 Encoder 那邊的英文序列做匹配，
        # 看當下字詞的 repr. 有多好並予以修正。
        # 記得我們一樣需要對 Encoder 的輸出套用 padding mask 避免關注到 <pad>
        attn2, attn_weights_block2 = self.mha2(enc_output, enc_output, out1, inp_padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)
        
        # sub-layer 3: FFN 部分跟 Encoder layer 完全一樣
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        
        ffn_output = self.dropout3(ffn_output, training=training)
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)
        
        # 除了主要輸出 `out3` 以外，輸出 multi-head 注意權重方便之後理解模型內部狀況
        return out3, attn_weights_block1, attn_weights_block2

# %%
# 實作 Encoder 裡頭主要包含了 3 個元件：輸入的詞嵌入層，位置編碼，N 個 Encoder layers
class Encoder(tf.keras.layers.Layer):
    # Encoder 的初始參數除了本來就要給 EncoderLayer 的參數還多了：
    # - num_layers: 決定要有幾個 EncoderLayers, 前面影片中的 `N`
    # - input_vocab_size: 用來把索引轉成詞嵌入向量
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, 
                 maximum_position_encoding, rate=0.1):
        super(Encoder, self).__init__()
        
        self.d_model = d_model
        
        # 注意 Encoder 已經包含了詞嵌入層 !!!
        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        
        # 建立 `num_layers` 個 EncoderLayers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)
    
    def call(self, x, training, mask):
        # 輸入的 x.shape == (batch_size, input_seq_len)
        # 以下各 layer 的輸出皆為 (batch_size, input_seq_len, d_model)
        input_seq_len = tf.shape(x)[1]
        
        # 將 2 維的索引序列轉成 3 維的詞嵌入張量，並依照論文乘上 sqrt(d_model)
        # 再加上對應長度的位置編碼
        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :input_seq_len, :]
        
        # 對 embedding 跟位置編碼的總合做 regularization
        # 這在 Decoder 也會做
        x = self.dropout(x, training=training)
        
        # 通過 N 個 EncoderLayer 做編碼
        for i, enc_layer in enumerate(self.enc_layers):
            x = enc_layer(x, training, mask)
        # 以下只是用來 demo EncoderLayer outputs
        #print('-' * 20)
        #print(f"EncoderLayer {i + 1}'s output:", x)
        return x

# 實作 Decoder 
class Decoder(tf.keras.layers.Layer):
    # 初始參數跟 Encoder 只差在用 `target_vocab_size` 而非 `inp_vocab_size`
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, 
                 maximum_position_encoding, rate=0.1):
        super(Decoder, self).__init__()
        
        self.d_model = d_model
        
        # 為中文（目標語言）建立詞嵌入層
        self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, self.d_model)
        
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        
        self.dropout = tf.keras.layers.Dropout(rate)
    
    # 呼叫時的參數跟 DecoderLayer 一模一樣
    def call(self, x, enc_output, training, combined_mask, inp_padding_mask):
        
        tar_seq_len = tf.shape(x)[1]
        attention_weights = {}  # 用來存放每個 Decoder layer 的注意權重
        
        # 這邊跟 Encoder 做的事情完全一樣
        x = self.embedding(x)  # (batch_size, tar_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :tar_seq_len, :]
        x = self.dropout(x, training=training)
        
        for i, dec_layer in enumerate(self.dec_layers):
            x, block1, block2 = dec_layer(x, enc_output, training, combined_mask, inp_padding_mask)
            
            # 將從每個 Decoder layer 取得的注意權重全部存下來回傳，方便我們觀察
            attention_weights['decoder_layer{}_block1'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2'.format(i + 1)] = block2
        
        # x.shape == (batch_size, tar_seq_len, d_model)
        return x, attention_weights

# %%
# 真正的 Transformer 模型需要 3 個元件: Encoder, Decoder, Final linear layer
# Transformer 之上已經沒有其他 layers 了，我們使用 tf.keras.Model 建立一個模型
# 輸入：1.英文序列：（batch_size, inp_seq_len）2.中文序列：（batch_size, tar_seq_len）
# 輸出：1.生成序列：（batch_size, tar_seq_len, target_vocab_size）2.注意權重的 dict
class Transformer(tf.keras.Model):
    # 初始參數包含 Encoder & Decoder 都需要超參數以及中英字典數目
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, 
                 pe_input, pe_target, rate=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, input_vocab_size, pe_input, rate)
        
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, target_vocab_size, pe_target, rate)
        
        # 這個 FFN 輸出跟中文字典一樣大的 logits 數，等通過 softmax 就代表每個中文字的出現機率
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)
    
    # enc_padding_mask 跟 dec_padding_mask 都是英文序列的 padding mask，
    # 只是一個給 Encoder layer 的 MHA 用，一個是給 Decoder layer 的 MHA 2 使用
    def call(self, inp, tar, training, enc_padding_mask, combined_mask, dec_padding_mask):
        
        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
        
        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(tar, enc_output, training, combined_mask, dec_padding_mask)
        
        # 將 Decoder 輸出通過最後一個 linear layer
        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
        
        # Transformer 每次的呼叫具體做的事情是：
        # 被輸入的多個 2 維英文張量 inp 會一路通過 Encoder 裡頭的詞嵌入層，
        # 位置編碼以及 N 個 Encoder layers 後被轉換成 Encoder 輸出 enc_output
        # 對應的輸入中文序列 tar 則會在 Decoder 裡頭走過相似的旅程並在每一層的 Decoder layer 
        # 利用 MHA 2 關注 Encoder 的輸出 enc_output，最後被 Decoder 輸出
        return final_output, attention_weights

# 實作位置編碼（Positional Encoding）直接參考 TensorFlow 官方 tutorial 
# 直觀的想法是想辦法讓被加入位置編碼的 word embedding 在 d_model 維度的空間裡頭不只會因為語義相近而靠近，
# 也會因為位置靠近而在該空間裡頭靠近
def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates

def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
    
    # apply sin to even indices in the array; 2i
    sines = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices in the array; 2i+1
    cosines = np.cos(angle_rads[:, 1::2])
    
    pos_encoding = np.concatenate([sines, cosines], axis=-1)
    
    pos_encoding = pos_encoding[np.newaxis, ...]
    # 第 1 維代表 batch_size，之後可以 broadcasting
    # 第 2 維是序列長度，我們會為每個在輸入 / 輸出序列裡頭的子詞都加入位置編碼
    # 第 3 維跟詞嵌入向量同維度
    return tf.cast(pos_encoding, dtype=tf.float32)

# 最後為 Transformer 的 Encoder / Decoder 準備遮罩
def create_masks(inp, tar):
    # 英文句子的 padding mask，要交給 Encoder layer 自注意力機制用的
    enc_padding_mask = create_padding_mask(inp)
    
    # 同樣也是英文句子的 padding mask，但是是要交給 Decoder layer 的 MHA 2 
    # 關注 Encoder 輸出序列用的
    dec_padding_mask = create_padding_mask(inp)
    
    # Decoder layer 的 MHA1 在做自注意力機制用的
    # `combined_mask` 是中文句子的 padding mask 跟 look ahead mask 的疊加
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)
    
    return enc_padding_mask, combined_mask, dec_padding_mask