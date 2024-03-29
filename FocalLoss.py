from functools import partial

import tensorflow as tf
import numbers

def check_type(obj, base, *, name=None, func=None, allow_none=False,
               default=None, error_message=None):
    """Check whether an object is an instance of a base type.
    Parameters
    ----------
    obj : object
        The object to be validated.
    name : str
        The name of `obj` in the calling function.
    base : type or tuple of type
        The base type that `obj` should be an instance of.
    func: callable, optional
        A function to be applied to `obj` if it is of type `base`. If None, no
        function will be applied and `obj` will be returned as-is.
    allow_none : bool, optional
        Indicates whether the value None should be allowed to pass through.
    default : object, optional
        The default value to return if `obj` is None and `allow_none` is True.
        If `default` is not None, it must be of type `base`, and it will have
        `func` applied to it if `func` is not None.
    error_message : str or None, optional
        Custom error message to display if the type is incorrect.
    Returns
    -------
    base type or None
        The validated object.
    Raises
    ------
    TypeError
        If `obj` is not an instance of `base`.
    Examples
    --------
    >>> check_type(1, int)
    1
    >>> check_type(1, (int, str))
    1
    >>> check_type(1, str)
    Traceback (most recent call last):
    ...
    TypeError: Invalid type. Expected: str. Actual: int.
    >>> check_type(1, (str, bool))
    Traceback (most recent call last):
    ...
    TypeError: Invalid type. Expected: (str, bool). Actual: int.
    >>> print(check_type(None, str, allow_none=True))
    None
    >>> check_type(1, str, name='num')
    Traceback (most recent call last):
    ...
    TypeError: Invalid type for parameter 'num'. Expected: str. Actual: int.
    >>> check_type(1, int, func=str)
    '1'
    >>> check_type(1, int, func='not callable')
    Traceback (most recent call last):
    ...
    ValueError: Parameter 'func' must be callable or None.
    >>> check_type(2.0, str, error_message='Not a string!')
    Traceback (most recent call last):
    ...
    TypeError: Not a string!
    >>> check_type(None, int, allow_none=True, default=0)
    0
    """
    if allow_none and obj is None:
        if default is not None:
            return check_type(default, base=base, name=name, func=func,
                              allow_none=False)
        return None

    if isinstance(obj, base):
        if func is None:
            return obj
        elif callable(func):
            return func(obj)
        else:
            raise ValueError('Parameter \'func\' must be callable or None.')

    # Handle wrong type
    if isinstance(base, tuple):
        expect = '(' + ', '.join(cls.__name__ for cls in base) + ')'
    else:
        expect = base.__name__
    actual = type(obj).__name__
    if error_message is None:
        error_message = 'Invalid type'
        if name is not None:
            error_message += f' for parameter \'{name}\''
        error_message += f'. Expected: {expect}. Actual: {actual}.'
    raise TypeError(error_message)

def check_bool(obj, *, name=None, allow_none=False, default=None):
    """Validate boolean function arguments.
    Parameters
    ----------
    obj : object
        The object to be validated.
    name : str, optional
        The name of `obj` in the calling function.
    allow_none : bool, optional
        Indicates whether the value None should be allowed.
    default : object, optional
        The default value to return if `obj` is None and `allow_none` is True.
    Returns
    -------
    bool or None
        The validated bool.
    Raises
    ------
    TypeError
        If `obj` is not an instance of bool.
    Examples
    --------
    >>> check_bool(True)
    True
    >>> check_bool(1.0)
    Traceback (most recent call last):
    ...
    TypeError: Invalid type. Expected: bool. Actual: float.
    >>> a = (1 < 2)
    >>> check_bool(a, name='a')
    True
    >>> b = 'not a bool'
    >>> check_bool(b, name='b')
    Traceback (most recent call last):
    ...
    TypeError: Invalid type for parameter 'b'. Expected: bool. Actual: str.
    """
    return check_type(obj, name=name, base=bool, func=bool,
                      allow_none=allow_none, default=default)
    
def _check_numeric(*, check_func, obj, name, base, func, positive, minimum,
                   maximum, allow_none, default):
    """Helper function for check_float and check_int."""
    obj = check_type(obj, name=name, base=base, func=func,
                     allow_none=allow_none, default=default)

    if obj is None:
        return None

    positive = check_bool(positive, name='positive')
    if positive and obj <= 0:
        if name is None:
            message = 'Parameter must be positive.'
        else:
            message = f'Parameter \'{name}\' must be positive.'
        raise ValueError(message)

    if minimum is not None:
        minimum = check_func(minimum, name='minimum')
        if obj < minimum:
            if name is None:
                message = f'Parameter must be at least {minimum}.'
            else:
                message = f'Parameter \'{name}\' must be at least {minimum}.'
            raise ValueError(message)

    if maximum is not None:
        maximum = check_func(maximum, name='minimum')
        if obj > maximum:
            if name is None:
                message = f'Parameter must be at most {maximum}.'
            else:
                message = f'Parameter \'{name}\' must be at most {maximum}.'
            raise ValueError(message)

    return obj

def check_float(obj, *, name=None, positive=False, minimum=None, maximum=None,
                allow_none=False, default=None):
    """Validate float function arguments.
    Parameters
    ----------
    obj : object
        The object to be validated.
    name : str, optional
        The name of `obj` in the calling function.
    positive : bool, optional
        Whether `obj` must be a positive float.
    minimum : float, optional
        The minimum value that `obj` can take (inclusive).
    maximum : float, optional
        The maximum value that `obj` can take (inclusive).
    allow_none : bool, optional
        Indicates whether the value None should be allowed.
    default : object, optional
        The default value to return if `obj` is None and `allow_none` is True.
    Returns
    -------
    float or None
        The validated float.
    Raises
    ------
    TypeError
        If `obj` is not a float.
    ValueError
        If any of the optional positivity or minimum and maximum value
        constraints are violated.
    Examples
    --------
    >>> check_float(0)
    0.0
    >>> check_float(1.0, positive=True)
    1.0
    >>> check_float(1.0 + 1.0j)
    Traceback (most recent call last):
    ...
    TypeError: Invalid type. Expected: Real. Actual: complex.
    >>> check_float(-1, positive=True)
    Traceback (most recent call last):
    ...
    ValueError: Parameter must be positive.
    >>> check_float(1.2, name='a', minimum=10)
    Traceback (most recent call last):
    ...
    ValueError: Parameter 'a' must be at least 10.0.
    """
    return _check_numeric(check_func=check_float, obj=obj, name=name,
                          base=numbers.Real, func=float, positive=positive,
                          minimum=minimum, maximum=maximum,
                          allow_none=allow_none, default=default)

_EPSILON = tf.keras.backend.epsilon()


def binary_focal_loss(y_true, y_pred, gamma, *, pos_weight=None,
                      from_logits=False, label_smoothing=None):
    r"""Focal loss function for binary classification.
    This loss function generalizes binary cross-entropy by introducing a
    hyperparameter :math:`\gamma` (gamma), called the *focusing parameter*,
    that allows hard-to-classify examples to be penalized more heavily relative
    to easy-to-classify examples.
    The focal loss [1]_ is defined as
    .. math::
        L(y, \hat{p})
        = -\alpha y \left(1 - \hat{p}\right)^\gamma \log(\hat{p})
        - (1 - y) \hat{p}^\gamma \log(1 - \hat{p})
    where
    *   :math:`y \in \{0, 1\}` is a binary class label,
    *   :math:`\hat{p} \in [0, 1]` is an estimate of the probability of the
        positive class,
    *   :math:`\gamma` is the *focusing parameter* that specifies how much
        higher-confidence correct predictions contribute to the overall loss
        (the higher the :math:`\gamma`, the higher the rate at which
        easy-to-classify examples are down-weighted).
    *   :math:`\alpha` is a hyperparameter that governs the trade-off between
        precision and recall by weighting errors for the positive class up or
        down (:math:`\alpha=1` is the default, which is the same as no
        weighting),
    The usual weighted binary cross-entropy loss is recovered by setting
    :math:`\gamma = 0`.
    Parameters
    ----------
    y_true : tensor-like
        Binary (0 or 1) class labels.
    y_pred : tensor-like
        Either probabilities for the positive class or logits for the positive
        class, depending on the `from_logits` parameter. The shapes of `y_true`
        and `y_pred` should be broadcastable.
    gamma : float
        The focusing parameter :math:`\gamma`. Higher values of `gamma` make
        easy-to-classify examples contribute less to the loss relative to
        hard-to-classify examples. Must be non-negative.
    pos_weight : float, optional
        The coefficient :math:`\alpha` to use on the positive examples. Must be
        non-negative.
    from_logits : bool, optional
        Whether `y_pred` contains logits or probabilities.
    label_smoothing : float, optional
        Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
        ground truth labels `y_true` are squeezed toward 0.5, with larger values
        of `label_smoothing` leading to label values closer to 0.5.
    Returns
    -------
    :class:`tf.Tensor`
        The focal loss for each example (assuming `y_true` and `y_pred` have the
        same shapes). In general, the shape of the output is the result of
        broadcasting the shapes of `y_true` and `y_pred`.
    Warnings
    --------
    This function does not reduce its output to a scalar, so it cannot be passed
    to :meth:`tf.keras.Model.compile` as a `loss` argument. Instead, use the
    wrapper class :class:`~focal_loss.BinaryFocalLoss`.
    Examples
    --------
    This function computes the per-example focal loss between a label and
    prediction tensor:
    >>> import numpy as np
    >>> from focal_loss import binary_focal_loss
    >>> loss = binary_focal_loss([0, 1, 1], [0.1, 0.7, 0.9], gamma=2)
    >>> np.set_printoptions(precision=3)
    >>> print(loss.numpy())
    [0.001 0.032 0.001]
    Below is a visualization of the focal loss between the positive class and
    predicted probabilities between 0 and 1. Note that as :math:`\gamma`
    increases, the losses for predictions closer to 1 get smoothly pushed to 0.
    .. plot::
        :include-source:
        :align: center
        import numpy as np
        import matplotlib.pyplot as plt
        from focal_loss import binary_focal_loss
        ps = np.linspace(0, 1, 100)
        gammas = (0, 0.5, 1, 2, 5)
        plt.figure()
        for gamma in gammas:
            loss = binary_focal_loss(1, ps, gamma=gamma)
            label = rf'$\gamma$={gamma}'
            if gamma == 0:
                label += ' (cross-entropy)'
            plt.plot(ps, loss, label=label)
        plt.legend(loc='best', frameon=True, shadow=True)
        plt.xlim(0, 1)
        plt.ylim(0, 4)
        plt.xlabel(r'Probability of positive class $\hat{p}$')
        plt.ylabel('Loss')
        plt.title(r'Plot of focal loss $L(1, \hat{p})$ for different $\gamma$',
                  fontsize=14)
        plt.show()
    Notes
    -----
    A classifier often estimates the positive class probability :math:`\hat{p}`
    by computing a real-valued *logit* :math:`\hat{y} \in \mathbb{R}` and
    applying the *sigmoid function* :math:`\sigma : \mathbb{R} \to (0, 1)`
    defined by
    .. math::
        \sigma(t) = \frac{1}{1 + e^{-t}}, \qquad (t \in \mathbb{R}).
    That is, :math:`\hat{p} = \sigma(\hat{y})`. In this case, the focal loss
    can be written as a function of the logit :math:`\hat{y}` instead of the
    predicted probability :math:`\hat{p}`:
    .. math::
        L(y, \hat{y})
        = -\alpha y \left(1 - \sigma(\hat{y})\right)^\gamma
        \log(\sigma(\hat{y}))
        - (1 - y) \sigma(\hat{y})^\gamma \log(1 - \sigma(\hat{y})).
    This is the formula that is computed when specifying `from_logits=True`.
    However, this formula is not very numerically stable if implemented
    directly; for example, there are multiple log and sigmoid computations
    involved. Instead, we use some tricks to rewrite it in the more numerically
    stable form
    .. math::
        L(y, \hat{y})
        = (1 - y) \hat{p}^\gamma \hat{y}
        + \left(\alpha y \hat{q}^\gamma + (1 - y) \hat{p}^\gamma\right)
        \left(\log(1 + e^{-|\hat{y}|}) + \max\{-\hat{y}, 0\}\right),
    where :math:`\hat{p} = \sigma(\hat{y})` and :math:`\hat{q} = 1 - \hat{p}`
    denote the estimates of the probabilities of the positive and negative
    classes, respectively.
    Indeed, starting with the observations that
    .. math::
        \log(\sigma(\hat{y}))
        = \log\left(\frac{1}{1 + e^{-\hat{y}}}\right)
        = -\log(1 + e^{-\hat{y}})
    and
    .. math::
        \log(1 - \sigma(\hat{y}))
        = \log\left(\frac{e^{-\hat{y}}}{1 + e^{-\hat{y}}}\right)
        = -\hat{y} - \log(1 + e^{-\hat{y}}),
    we obtain
    .. math::
        \begin{aligned}
        L(y, \hat{y})
        &= -\alpha y \hat{q}^\gamma \log(\sigma(\hat{y}))
        - (1 - y) \hat{p}^\gamma \log(1 - \sigma(\hat{y})) \\
        &= \alpha y \hat{q}^\gamma \log(1 + e^{-\hat{y}})
        + (1 - y) \hat{p}^\gamma \left(\hat{y} + \log(1 + e^{-\hat{y}})\right)\\
        &= (1 - y) \hat{p}^\gamma \hat{y}
        + \left(\alpha y \hat{q}^\gamma + (1 - y) \hat{p}^\gamma\right)
        \log(1 + e^{-\hat{y}}).
        \end{aligned}
    Note that if :math:`\hat{y} < 0`, then the exponential term
    :math:`e^{-\hat{y}}` could become very large. In this case, we can instead
    observe that
    .. math::
        \begin{align*}
        \log(1 + e^{-\hat{y}})
        &= \log(1 + e^{-\hat{y}}) + \hat{y} - \hat{y} \\
        &= \log(1 + e^{-\hat{y}}) + \log(e^{\hat{y}}) - \hat{y} \\
        &= \log(1 + e^{\hat{y}}) - \hat{y}.
        \end{align*}
    Moreover, the :math:`\hat{y} < 0` and :math:`\hat{y} \geq 0` cases can be
    unified by writing
    .. math::
        \log(1 + e^{-\hat{y}})
        = \log(1 + e^{-|\hat{y}|}) + \max\{-\hat{y}, 0\}.
    Thus, we arrive at the numerically stable formula shown earlier.
    References
    ----------
    .. [1] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár. Focal loss for
        dense object detection. IEEE Transactions on Pattern Analysis and
        Machine Intelligence, 2018.
        (`DOI <https://doi.org/10.1109/TPAMI.2018.2858826>`__)
        (`arXiv preprint <https://arxiv.org/abs/1708.02002>`__)
    See Also
    --------
    :meth:`~focal_loss.BinaryFocalLoss`
        A wrapper around this function that makes it a
        :class:`tf.keras.losses.Loss`.
    """
    # Validate arguments
    gamma = check_float(gamma, name='gamma', minimum=0)
    pos_weight = check_float(pos_weight, name='pos_weight', minimum=0,
                             allow_none=True)
    from_logits = check_bool(from_logits, name='from_logits')
    label_smoothing = check_float(label_smoothing, name='label_smoothing',
                                  minimum=0, maximum=1, allow_none=True)

    # Ensure predictions are a floating point tensor; converting labels to a
    # tensor will be done in the helper functions
    y_pred = tf.convert_to_tensor(y_pred)
    if not y_pred.dtype.is_floating:
        y_pred = tf.dtypes.cast(y_pred, dtype=tf.float32)

    # Delegate per-example loss computation to helpers depending on whether
    # predictions are logits or probabilities
    if from_logits:
        return _binary_focal_loss_from_logits(labels=y_true, logits=y_pred,
                                              gamma=gamma,
                                              pos_weight=pos_weight,
                                              label_smoothing=label_smoothing)
    else:
        return _binary_focal_loss_from_probs(labels=y_true, p=y_pred,
                                             gamma=gamma, pos_weight=pos_weight,
                                             label_smoothing=label_smoothing)

@tf.keras.utils.register_keras_serializable()
class BinaryFocalLoss(tf.keras.losses.Loss):
    r"""Focal loss function for binary classification.
    This loss function generalizes binary cross-entropy by introducing a
    hyperparameter called the *focusing parameter* that allows hard-to-classify
    examples to be penalized more heavily relative to easy-to-classify examples.
    This class is a wrapper around :class:`~focal_loss.binary_focal_loss`. See
    the documentation there for details about this loss function.
    Parameters
    ----------
    gamma : float
        The focusing parameter :math:`\gamma`. Must be non-negative.
    pos_weight : float, optional
        The coefficient :math:`\alpha` to use on the positive examples. Must be
        non-negative.
    from_logits : bool, optional
        Whether model prediction will be logits or probabilities.
    label_smoothing : float, optional
        Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
        ground truth labels are squeezed toward 0.5, with larger values of
        `label_smoothing` leading to label values closer to 0.5.
    **kwargs : keyword arguments
        Other keyword arguments for :class:`tf.keras.losses.Loss` (e.g., `name`
        or `reduction`).
    Examples
    --------
    An instance of this class is a callable that takes a tensor of binary ground
    truth labels `y_true` and a tensor of model predictions `y_pred` and returns
    a scalar tensor obtained by reducing the per-example focal loss (the default
    reduction is a batch-wise average).
    >>> from focal_loss import BinaryFocalLoss
    >>> loss_func = BinaryFocalLoss(gamma=2)
    >>> loss = loss_func([0, 1, 1], [0.1, 0.7, 0.9])  # A scalar tensor
    >>> print(f'Mean focal loss: {loss.numpy():.3f}')
    Mean focal loss: 0.011
    Use this class in the :mod:`tf.keras` API like any other binary
    classification loss function class found in :mod:`tf.keras.losses` (e.g.,
    :class:`tf.keras.losses.BinaryCrossentropy`:
    .. code-block:: python
        # Typical usage
        model = tf.keras.Model(...)
        model.compile(
            optimizer=...,
            loss=BinaryFocalLoss(gamma=2),  # Used here like a tf.keras loss
            metrics=...,
        )
        history = model.fit(...)
    See Also
    --------
    :meth:`~focal_loss.binary_focal_loss`
        The function that performs the focal loss computation, taking a label
        tensor and a prediction tensor and outputting a loss.
    """

    def __init__(self, gamma, *, pos_weight=None, from_logits=False,
                 label_smoothing=None, **kwargs):
        # Validate arguments
        gamma = check_float(gamma, name='gamma', minimum=0)
        pos_weight = check_float(pos_weight, name='pos_weight', minimum=0,
                                 allow_none=True)
        from_logits = check_bool(from_logits, name='from_logits')
        label_smoothing = check_float(label_smoothing, name='label_smoothing',
                                      minimum=0, maximum=1, allow_none=True)

        super().__init__(**kwargs)
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def get_config(self):
        """Returns the config of the layer.
        A layer config is a Python dictionary containing the configuration of a
        layer. The same layer can be re-instantiated later (without its trained
        weights) from this configuration.
        Returns
        -------
        dict
            This layer's config.
        """
        config = super().get_config()
        config.update(gamma=self.gamma, pos_weight=self.pos_weight,
                      from_logits=self.from_logits,
                      label_smoothing=self.label_smoothing)
        return config

    def call(self, y_true, y_pred):
        """Compute the per-example focal loss.
        This method simply calls :meth:`~focal_loss.binary_focal_loss` with the
        appropriate arguments.
        Parameters
        ----------
        y_true : tensor-like
            Binary (0 or 1) class labels.
        y_pred : tensor-like
            Either probabilities for the positive class or logits for the
            positive class, depending on the `from_logits` attribute. The shapes
            of `y_true` and `y_pred` should be broadcastable.
        Returns
        -------
        :class:`tf.Tensor`
            The per-example focal loss. Reduction to a scalar is handled by
            this layer's :meth:`~focal_loss.BinaryFocalLoss.__call__` method.
        """
        return binary_focal_loss(y_true=y_true, y_pred=y_pred, gamma=self.gamma,
                                 pos_weight=self.pos_weight,
                                 from_logits=self.from_logits,
                                 label_smoothing=self.label_smoothing)

def _process_labels(labels, label_smoothing, dtype):
    """Pre-process a binary label tensor, maybe applying smoothing.
    Parameters
    ----------
    labels : tensor-like
        Tensor of 0's and 1's.
    label_smoothing : float or None
        Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
        ground truth labels `y_true` are squeezed toward 0.5, with larger values
        of `label_smoothing` leading to label values closer to 0.5.
    dtype : tf.dtypes.DType
        Desired type of the elements of `labels`.
    Returns
    -------
    tf.Tensor
        The processed labels.
    """
    labels = tf.dtypes.cast(labels, dtype=dtype)
    if label_smoothing is not None:
        labels = (1 - label_smoothing) * labels + label_smoothing * 0.5
    return labels

def _binary_focal_loss_from_logits(labels, logits, gamma, pos_weight,
                                   label_smoothing):
    """Compute focal loss from logits using a numerically stable formula.
    Parameters
    ----------
    labels : tensor-like
        Tensor of 0's and 1's: binary class labels.
    logits : tf.Tensor
        Logits for the positive class.
    gamma : float
        Focusing parameter.
    pos_weight : float or None
        If not None, losses for the positive class will be scaled by this
        weight.
    label_smoothing : float or None
        Float in [0, 1]. When 0, no smoothing occurs. When positive, the binary
        ground truth labels `y_true` are squeezed toward 0.5, with larger values
        of `label_smoothing` leading to label values closer to 0.5.
    Returns
    -------
    tf.Tensor
        The loss for each example.
    """
    labels = _process_labels(labels=labels, label_smoothing=label_smoothing,
                             dtype=logits.dtype)

    # Compute probabilities for the positive class
    p = tf.math.sigmoid(logits)

    # Without label smoothing we can use TensorFlow's built-in per-example cross
    # entropy loss functions and multiply the result by the modulating factor.
    # Otherwise, we compute the focal loss ourselves using a numerically stable
    # formula below
    if label_smoothing is None:
        # The labels and logits tensors' shapes need to be the same for the
        # built-in cross-entropy functions. Since we want to allow broadcasting,
        # we do some checks on the shapes and possibly broadcast explicitly
        # Note: tensor.shape returns a tf.TensorShape, whereas tf.shape(tensor)
        # returns an int tf.Tensor; this is why both are used below
        labels_shape = labels.shape
        logits_shape = logits.shape
        if not labels_shape.is_fully_defined() or labels_shape != logits_shape:
            labels_shape = tf.shape(labels)
            logits_shape = tf.shape(logits)
            shape = tf.broadcast_dynamic_shape(labels_shape, logits_shape)
            labels = tf.broadcast_to(labels, shape)
            logits = tf.broadcast_to(logits, shape)
        if pos_weight is None:
            loss_func = tf.nn.sigmoid_cross_entropy_with_logits
        else:
            loss_func = partial(tf.nn.weighted_cross_entropy_with_logits,
                                pos_weight=pos_weight)
        loss = loss_func(labels=labels, logits=logits)
        modulation_pos = (1 - p) ** gamma
        modulation_neg = p ** gamma
        mask = tf.dtypes.cast(labels, dtype=tf.bool)
        modulation = tf.where(mask, modulation_pos, modulation_neg)
        return modulation * loss

    # Terms for the positive and negative class components of the loss
    pos_term = labels * ((1 - p) ** gamma)
    neg_term = (1 - labels) * (p ** gamma)

    # Term involving the log and ReLU
    log_weight = pos_term
    if pos_weight is not None:
        log_weight *= pos_weight
    log_weight += neg_term
    log_term = tf.math.log1p(tf.math.exp(-tf.math.abs(logits)))
    log_term += tf.nn.relu(-logits)
    log_term *= log_weight

    # Combine all the terms into the loss
    loss = neg_term * logits + log_term
    return loss
