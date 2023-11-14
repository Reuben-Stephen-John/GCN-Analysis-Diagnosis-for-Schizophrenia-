# first line: 536
@fill_doc
def clean(
    signals,
    runs=None,
    detrend=True,
    standardize="zscore",
    sample_mask=None,
    confounds=None,
    standardize_confounds=True,
    filter="butterworth",
    low_pass=None,
    high_pass=None,
    t_r=2.5,
    ensure_finite=False,
    **kwargs,
):
    """Improve :term:`SNR` on masked :term:`fMRI` signals.

    This function can do several things on the input signals. With the default
    options, the procedures are performed in the following order:

    - detrend
    - low- and high-pass butterworth filter
    - remove confounds
    - standardize

    Low-pass filtering improves specificity.

    High-pass filtering should be kept small, to keep some sensitivity.

    Butterworth filtering is only meaningful on evenly-sampled signals.

    When performing scrubbing (censoring high-motion volumes) with butterworth
    filtering, the signal is processed in the following order, based on the
    second recommendation in :footcite:`Lindquist2018`:

    - interpolate high motion volumes with cubic spline interpolation
    - detrend
    - low- and high-pass butterworth filter
    - censor high motion volumes
    - remove confounds
    - standardize

    According to :footcite:`Lindquist2018`, removal of confounds will be done
    orthogonally to temporal filters (low- and/or high-pass filters), if both
    are specified. The censored volumes should be removed in both signals and
    confounds before the nuisance regression.

    When performing scrubbing with cosine drift term filtering, the signal is
    processed in the following order, based on the first recommendation in
    :footcite:`Lindquist2018`:

    - generate cosine drift term
    - censor high motion volumes in both signal and confounds
    - detrend
    - remove confounds
    - standardize

    Parameters
    ----------
    signals : :class:`numpy.ndarray`
        Timeseries. Must have shape (instant number, features number).
        This array is not modified.

    runs : :class:`numpy.ndarray`, optional
        Add a run level to the cleaning process. Each run will be
        cleaned independently. Must be a 1D array of n_samples elements.
        Default is None.

    confounds : :class:`numpy.ndarray`, :obj:`str`, :class:`pathlib.Path`,\
    :class:`pandas.DataFrame` or :obj:`list` of confounds timeseries.
        Shape must be (instant number, confound number), or just
        (instant number,).
        The number of time instants in ``signals`` and ``confounds`` must be
        identical (i.e. ``signals.shape[0] == confounds.shape[0]``).
        If a string is provided, it is assumed to be the name of a csv file
        containing signals as columns, with an optional one-line header.
        If a list is provided, all confounds are removed from the input
        signal, as if all were in the same array.
        Default is None.

    sample_mask : None, Any type compatible with numpy-array indexing, \
        or :obj:`list` of
        shape: (number of scans - number of volumes removed, ) for explicit \
            index, or (number of scans, ) for binary mask
        Masks the niimgs along time/fourth dimension to perform scrubbing
        (remove volumes with high motion) and/or non-steady-state volumes.
        When passing binary mask with boolean values, ``True`` refers to
        volumes kept, and ``False`` for volumes removed.
        This masking step is applied before signal cleaning. When supplying run
        information, sample_mask must be a list containing sets of indexes for
        each run.

            .. versionadded:: 0.8.0

        Default is None.
    %(t_r)s
        Default=2.5.
    filter : {'butterworth', 'cosine', False}, optional
        Filtering methods:

            - 'butterworth': perform butterworth filtering.
            - 'cosine': generate discrete cosine transformation drift terms.
            - False: Do not perform filtering.

        Default='butterworth'.
    %(low_pass)s

        .. note::
            `low_pass` is not implemented for filter='cosine'.

    %(high_pass)s
    %(detrend)s
    standardize : {'zscore_sample', 'zscore', 'psc', True, False}, optional
        Strategy to standardize the signal:

            - 'zscore_sample': The signal is z-scored. Timeseries are shifted
              to zero mean and scaled to unit variance. Uses sample std.
            - 'zscore': The signal is z-scored. Timeseries are shifted
              to zero mean and scaled to unit variance. Uses population std
              by calling default :obj:`numpy.std` with N - ``ddof=0``.
            - 'psc':  Timeseries are shifted to zero mean value and scaled
              to percent signal change (as compared to original mean signal).
            - True: The signal is z-scored (same as option `zscore`).
              Timeseries are shifted to zero mean and scaled to unit variance.
            - False: Do not standardize the data.

        Default="zscore".
    %(standardize_confounds)s

    ensure_finite : :obj:`bool`, default=False
        If `True`, the non-finite values (NANs and infs) found in the data
        will be replaced by zeros.

    kwargs : dict
        Keyword arguments to be passed to functions called within ``clean``.
        Kwargs prefixed with ``'butterworth__'`` will be passed to
        :func:`~nilearn.signal.butterworth`.


    Returns
    -------
    cleaned_signals : :class:`numpy.ndarray`
        Input signals, cleaned. Same shape as `signals` unless `sample_mask`
        is applied.

    Notes
    -----
    Confounds removal is based on a projection on the orthogonal
    of the signal space. See :footcite:`Friston1994`.

    Orthogonalization between temporal filters and confound removal is based on
    suggestions in :footcite:`Lindquist2018`.

    References
    ----------
    .. footbibliography::

    See Also
    --------
    nilearn.image.clean_img
    """
    # Raise warning for some parameter combinations when confounds present
    confounds = stringify_path(confounds)
    if confounds is not None:
        _check_signal_parameters(detrend, standardize_confounds)
    # check if filter parameters are satisfied and return correct filter
    filter_type = _check_filter_parameters(filter, low_pass, high_pass, t_r)

    # Read confounds and signals
    signals, runs, confounds, sample_mask = _sanitize_inputs(
        signals, runs, confounds, sample_mask, ensure_finite
    )

    # Process each run independently
    if runs is not None:
        return _process_runs(
            signals,
            runs,
            detrend,
            standardize,
            confounds,
            sample_mask,
            filter_type,
            low_pass,
            high_pass,
            t_r,
        )

    # For the following steps, sample_mask should be either None or index-like

    # Generate cosine drift terms using the full length of the signals
    if filter_type == "cosine":
        confounds = _create_cosine_drift_terms(
            signals, confounds, high_pass, t_r
        )

    # Interpolation / censoring
    signals, confounds = _handle_scrubbed_volumes(
        signals, confounds, sample_mask, filter_type, t_r
    )

    # Detrend
    # Detrend and filtering should apply to confounds, if confound presents
    # keep filters orthogonal (according to Lindquist et al. (2018))
    # Restrict the signal to the orthogonal of the confounds
    if detrend:
        mean_signals = signals.mean(axis=0)
        signals = _standardize(signals, standardize=False, detrend=detrend)
        if confounds is not None:
            confounds = _standardize(
                confounds, standardize=False, detrend=detrend
            )

    # Butterworth filtering
    if filter_type == "butterworth":
        butterworth_kwargs = {
            k.replace("butterworth__", ""): v
            for k, v in kwargs.items()
            if k.startswith("butterworth__")
        }
        signals = butterworth(
            signals,
            sampling_rate=1.0 / t_r,
            low_pass=low_pass,
            high_pass=high_pass,
            **butterworth_kwargs,
        )
        if confounds is not None:
            # Apply low- and high-pass filters to keep filters orthogonal
            # (according to Lindquist et al. (2018))
            confounds = butterworth(
                confounds,
                sampling_rate=1.0 / t_r,
                low_pass=low_pass,
                high_pass=high_pass,
                **butterworth_kwargs,
            )

        # apply sample_mask to remove censored volumes after signal filtering
        if sample_mask is not None:
            signals, confounds = _censor_signals(
                signals, confounds, sample_mask
            )

    # Remove confounds
    if confounds is not None:
        confounds = _standardize(
            confounds, standardize=standardize_confounds, detrend=False
        )
        if not standardize_confounds:
            # Improve numerical stability by controlling the range of
            # confounds. We don't rely on _standardize as it removes any
            # constant contribution to confounds.
            confound_max = np.max(np.abs(confounds), axis=0)
            confound_max[confound_max == 0] = 1
            confounds /= confound_max

        # Pivoting in qr decomposition was added in scipy 0.10
        Q, R, _ = linalg.qr(confounds, mode="economic", pivoting=True)
        Q = Q[:, np.abs(np.diag(R)) > np.finfo(np.float64).eps * 100.0]
        signals -= Q.dot(Q.T).dot(signals)

    # Standardize
    if detrend and (standardize == "psc"):
        # If the signal is detrended, we have to know the original mean
        # signal to calculate the psc.
        signals = _standardize(
            signals + mean_signals, standardize=standardize, detrend=False
        )
    else:
        signals = _standardize(signals, standardize=standardize, detrend=False)

    return signals
