import logging
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Shared architecture (TripleNet: phi-Net, n-Net, p-Net, Tanh MLPs) ----
hiddim = 128
n_layers = 4
film_hidden = 32  # hidden width of each subnetwork's FiLM hypernetwork MLP (FiLM is switched on
                  # by passing use_film=True to TripleNet, not by a config flag)

# ---- Shared training / sampling ----
centered_coords = True  # centre chi on the junction so tanh and the junction line up
qfl_v_centre = 12.361324787139893  # fixed V/Ut centring of the FiLM voltage column (the paper's value, used for every device set)
boundary_trim_nodes = 1  # raw DEVSIM nodes removed at each contact (= 1 nm)
use_trimmed_domain = False  # drop DEVSIM's last (orders-of-magnitude-jumping) mesh cell at each contact
grad_clip = 1.0
lr_gamma = 0.1
log_every = 200
checkpoint_every = 2000
n_interior = 4096    # SamplingConfig default: interior collocation points per anchor
resample_every = 400  # SamplingConfig default: resample collocation points every N epochs

# ---- QFL pipeline (v_n_hat / v_p_hat quasi-Fermi-level space) ----
qfl_lr_init = 0.001
qfl_lr_step = 7000
qfl_n_epochs = 20000
qfl_seed = int(os.environ.get("QFL_SEED", 0))
qfl_minibatch_size = 64  # anchors per epoch - a full 480-990 anchor pool does not fit an 8GB GPU
qfl_n_interior = 256  # collocation points per anchor per epoch
qfl_clamp = 15.0  # cond_qfl's rescaled-coordinate clamp

# Reference doping Cref every density is divided by ("n_max" in the paper): max(nA, nD), equal to nD for a symmetric device
# (devsim_source.reference_doping accepts "max" | "min" | "geomean").
cref_mode = "max"

# ---- 2D (src/source2d.py devices) ----
qfl2d_extrude_aspect = 0.5  # Extruded1DSource height as a fraction of L
qfl2d_junction_frac = 0.25  # share of collocation/data points concentrated near the metallurgical junction |C| ~ 0
qfl2d_n_bc_per_contact = 32  # Dirichlet points per contact segment
qfl2d_n_neumann = 128  # points per anchor per epoch on the insulating boundary
lambda_neumann = 1.0  # weight of the insulating-boundary (zero normal field / current) loss

# QFL FiLM recipe (src/run_train_eval_qfl_film.py): FiLM-conditioned model on cond_qfl_final_physical_film with the
# adaptive clamp. No eta conditioning column, no SRH-peak continuity divisor, no symmetric-BC pinning.
qfl_film_phi_net_hiddim = 128  # phi_net is wider than n_net/p_net (config.hiddim)
qfl_film_normalize_by_mobility = True
qfl_film_nD_ref_for_diffusion = 1e15  # pinned reference of the Neumann-loss current normalizer
qfl_measurable_i_dev = 1e-6  # A/cm2: below this |I_devsim| the current ratio is noise, not scored
qfl_collapse_ratio = 0.05  # measurable anchors with |I_net/I_dev| below this count as collapsed

# ---- Output ----
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)


def setup_logging(level=logging.INFO):
    """Configure root logging once; every module uses logging.getLogger(__name__)
    and inherits it. Also appends to results_dir/run.log."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(results_dir, "run.log"), mode="a")
        ]
    )
