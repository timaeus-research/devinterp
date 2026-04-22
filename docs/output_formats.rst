.. _output-formats:

Output Formats
==============

All devinterp functions return xarray objects backed by Zarr. This page
documents the exact structure of each output.

Working with Zarr Output
------------------------

``sample()`` writes results to a ``.zarr`` directory (by default in a temp
directory, or at ``output_path`` if specified). The returned ``xr.DataTree``
is lazy-loaded — data is read from disk on access.

.. code-block:: python

   # Save to a specific path
   tree = sample(..., output_path="my_experiment.zarr")

   # Reopen later
   import xarray as xr
   tree = xr.open_datatree("my_experiment.zarr", engine="zarr", consolidated=False)

   # Access data
   loss = tree.dataset["sampling_loss"]  # lazy xr.DataArray
   loss.values  # loads into memory as numpy array

   # Post-process from saved results
   from devinterp.slt.llc import compute_llc
   result = compute_llc(tree)


``sample()`` → ``xr.DataTree``
------------------------------

The root dataset contains all arrays at the top level.

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Variable
     - Dimensions
     - Description
   * - ``init_loss``
     - (chain, token_pos)
     - Mean per-token loss before sampling, averaged over ``num_init_loss_batches`` batches
   * - ``sampling_loss``
     - (chain, draw, batch, token_pos)
     - Per-token loss on the sampling dataset at each draw
   * - ``loss_{obs}``
     - (chain, draw, batch_{obs}, token_pos)
     - Per-token loss on observable ``obs`` at each draw
   * - ``input_ids_{obs}``
     - (batch_{obs}, token)
     - Fixed input IDs for observable ``obs`` (same across all draws)
   * - ``n_beta``
     - scalar
     - Inverse temperature used for sampling

When ``save_metrics=True``, additional per-step SGLD diagnostics are included:

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Variable
     - Dimensions
     - Description
   * - ``metrics_scaled_grad``
     - (chain, step)
     - L2 norm of the scaled gradient component
   * - ``metrics_unscaled_grad``
     - (chain, step)
     - L2 norm of the unscaled (raw) gradient
   * - ``metrics_localization``
     - (chain, step)
     - L2 norm of the localization force
   * - ``metrics_weight_decay``
     - (chain, step)
     - L2 norm of the weight decay component
   * - ``metrics_noise``
     - (chain, step)
     - L2 norm of the injected noise
   * - ``metrics_distance``
     - (chain, step)
     - L2 distance from initial parameters
   * - ``metrics_dot_grad_prior``
     - (chain, step)
     - Dot product of gradient and prior components
   * - ``metrics_dot_grad_noise``
     - (chain, step)
     - Dot product of gradient and noise
   * - ``metrics_dot_prior_noise``
     - (chain, step)
     - Dot product of prior and noise

Where ``step = num_burnin_steps + num_draws * num_steps_bw_draws``.

Metadata is stored in ``tree.attrs["metadata"]`` as a nested dict containing
the ``SamplerConfig``, observable specs, and other configuration.


``compute_llc()`` → ``xr.Dataset``
-----------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 20 55

   * - Variable
     - Dimensions
     - Description
   * - ``llc_mean``
     - scalar
     - Mean LLC across chains: ``mean(llc_per_chain)``
   * - ``llc_std``
     - scalar
     - Std of LLC across chains
   * - ``llc_per_chain``
     - (chain,)
     - LLC per chain: ``n_beta * (mean_loss_per_chain - init_loss)``
   * - ``llc_scalar``
     - scalar
     - LLC with all dims reduced at once (for bitwise parity with aether)
   * - ``loss_trace``
     - (chain, draw)
     - Mean loss per chain per draw (averaged over batch and token_pos)
   * - ``init_loss``
     - scalar
     - Mean init loss (averaged over all dims)


``compute_bif()`` → ``xr.Dataset``
------------------------------------

With ``correlation_method="token"`` (default):

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Variable
     - Dimensions
     - Description
   * - ``influences``
     - (batch_1, batch_2, target_position, target_position_T)
     - Token-wise pairwise correlation matrix between all sequence pairs.
       With ``average_tokenwise_bif=True``, collapses to (batch_1, batch_2).
   * - ``input_ids``
     - (batch, position)
     - Concatenated input IDs from all observables

With ``correlation_method="sequence"``:

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Variable
     - Dimensions
     - Description
   * - ``influences``
     - (batch_1, batch_2)
     - Sequence-level pairwise correlation (loss averaged over tokens first)
   * - ``input_ids``
     - (batch, position)
     - Concatenated input IDs from all observables

Coordinates ``batch_1`` and ``batch_2`` are integer indices into the
concatenated observable sequences. ``target_position`` is 1-indexed.


``compute_susceptibilities()`` → ``xr.DataTree``
--------------------------------------------------

.. code-block:: text

   /susceptibilities
       sus            (sus_flat, wr)     — susceptibility values
       Coordinates:
           wr         (wr,)              — weight restriction names (excluding "full")
           dataset_id (sus_flat,)        — which observable each entry belongs to
           batch      (sus_flat,)        — batch index within that observable
           target_position (sus_flat,)   — token position (1-indexed)
       Attributes:
           dataset_id_to_name            — list mapping dataset_id integers to names

   /context
       input_ids      (ctx_flat,)        — flattened input token IDs
       Coordinates:
           dataset_id (ctx_flat,)        — which observable
           batch      (ctx_flat,)        — batch index
           position   (ctx_flat,)        — token position (0-indexed, includes BOS)
       Attributes:
           dataset_id_to_name            — same mapping as /susceptibilities

The ``sus_flat`` dimension flattens ``(batch, target_position)`` across all
observables. Use the ``dataset_id``, ``batch``, and ``target_position``
coordinates to reconstruct the original structure.
