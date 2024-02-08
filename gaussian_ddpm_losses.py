import torch
import numpy as np
import torch.nn.functional as F
import math

def sum_except_batch(x, num_dims=1):
    return torch.sum(x, dim = -1)


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s = 0.015):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    timesteps = (
        torch.arange(timesteps + 1, dtype=torch.float64) / timesteps + s
    )
    alphas = timesteps / (1 + s) * math.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = betas.clamp(max=0.999)
    betas = torch.cat(
            (torch.tensor([0], dtype=torch.float64), betas), 0
        )
    betas = betas.clamp(min=0.001)
    return betas


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.000004
    beta_end = scale * 0.0008
    betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)
    betas = torch.cat((torch.tensor([0.0001], dtype=torch.float64), betas), 0)
    return betas


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    Compute the KL divergence between two gaussians.

    Shapes are automatically broadcasted, so batches can be compared to
    scalars, among other use cases.
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # Force variances to be Tensors. Broadcasting helps convert scalars to
    # Tensors, but it does not work for th.exp().
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )

def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    Compute the log-likelihood of a Gaussian distribution discretizing to a
    given image.

    :param x: the target images. It is assumed that this was uint8 values,
              rescaled to the range [-1, 1].
    :param means: the Gaussian mean Tensor.
    :param log_scales: the Gaussian log stddev Tensor.
    :return: a tensor like x of log probabilities (in nats).
    """
    assert x.shape == means.shape == log_scales.shape
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    cdf_plus = approx_standard_normal_cdf(plus_in)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)
    cdf_min = approx_standard_normal_cdf(min_in)
    log_cdf_plus = torch.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = torch.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min
    log_probs = torch.where(
        x < -0.999,
        log_cdf_plus,
        torch.where(x > 0.999, log_one_minus_cdf_min, torch.log(cdf_delta.clamp(min=1e-12))),
    )
    assert log_probs.shape == x.shape
    return log_probs

def approx_standard_normal_cdf(x):
    """
    A fast approximation of the cumulative distribution function of the
    standard normal.
    """
    return 0.5 * (1.0 + torch.tanh(np.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

class diffusion_model(torch.nn.Module):
    def __init__(self, timesteps):
        super(diffusion_model, self).__init__()

        betas = linear_beta_schedule(timesteps)
        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat((torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        device = torch.device('cuda', torch.distributed.get_rank())


        self.register("posterior_variance", posterior_variance.to(device))
        self.register("betas", betas.to(device))
        self.register("alphas", alphas.to(device))
        self.register("alphas_cumprod", alphas_cumprod.to(device))
        self.register("sqrt_alphas", torch.sqrt(alphas).to(device))
        self.register("alphas_cumprod_prev", alphas_cumprod_prev.to(device))
        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(device))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod).to(device))

        self.register("thresh", (1-self.alphas)/self.sqrt_one_minus_alphas_cumprod)
        self.register("posterior_log_variance_clipped", torch.log(self.posterior_variance))
        self.register("posterior_mean_coef1", self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.register("posterior_mean_coef2", (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod))
        self.register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / self.alphas_cumprod - 1))
        self.num_timesteps = timesteps
        self.device = device


    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def q_sample(self, x, t):
        noise = torch.randn_like(x)
        return (
            self.sqrt_alphas_cumprod[t] * x
            + self.sqrt_one_minus_alphas_cumprod[t] * noise, noise
        )

    def q_sample_inter(self, x, t, k):
        noise = torch.randn_like(x)
        var = torch.sqrt(1-self.alphas_cumprod[t+k]/self.alphas_cumprod[t])
        return (
            self.sqrt_alphas_cumprod[t+k] / self.sqrt_alphas_cumprod[t] * x
            + var * noise
        )

    def q_posterior_mean_variance(self, x_0, x_t, t):
        """Compute the mean and variance of the diffusion posterior: q(x_{t-1} | x_t, x_0)"""
        posterior_mean = self.posterior_mean_coef1[t] * x_0 + self.posterior_mean_coef2[t] * x_t
        posterior_var = self.posterior_variance[t]*torch.ones_like(x_0)
        log_var_clipped = self.posterior_log_variance_clipped[t]*torch.ones_like(x_0)

        return posterior_mean, posterior_var, log_var_clipped

    def p_mean_variance(self, model_output, x, t, clip_denoised=True, denoised_fn=None, model_kwargs=None):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        H, C = x.shape[:2]

        model_output, model_var_values = torch.split(model_output, C, dim=1)
        min_log = self.posterior_log_variance_clipped[t]
        max_log = torch.log(self.betas[t])
        frac = (model_var_values + 1) / 2
        model_log_variance = frac * max_log + (1 - frac) * min_log
        model_variance = torch.exp(model_log_variance)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        pred_xstart = process_xstart(self.predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
        model_mean, _, _ = self.q_posterior_mean_variance(x_0=pred_xstart, x_t=x, t=t)

        return {
            "mean": model_mean,
            "variance": model_variance,
            "log_variance": model_log_variance,
            "pred_xstart": pred_xstart,
            "model_output": model_output
        }

    def predict_xstart_from_eps(self, x_t, t, eps):
        x_0 = self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * eps
        return x_0

    def compute_vb_loss(self, pred_y, x_0, x_t, t):
        true_mean, _, true_log_variances = self.q_posterior_mean_variance(x_0, x_t, t)
        out = self.p_mean_variance(pred_y, x_t, t)
        kl = normal_kl(true_mean, true_log_variances, out["mean"], out["log_variance"])
        kl = torch.mean(kl) / torch.log(torch.tensor(2.0, dtype=torch.float64).to(self.device))
        decoder_nll = -discretized_gaussian_log_likelihood(x=out["pred_xstart"], means=out["mean"], log_scales=0.5 * out["log_variance"])
        assert decoder_nll.shape == out["pred_xstart"].shape
        decoder_nll = torch.mean(decoder_nll) / torch.log(torch.tensor(2.0, dtype=torch.float64).to(self.device))
        output = torch.where((t == 0), decoder_nll, kl)

        return output

class gaussian_ddpm_losses():
    def __init__(self, num_timesteps, device, unweighted_MSE, time_batch=1):
        self.diff_Y = diffusion_model(timesteps=num_timesteps)
        self.num_timesteps = num_timesteps
        self.device = device
        self.time_batch = time_batch
        self.unweighted_MSE = unweighted_MSE
            
    def loss_fn(self, model, x, adj, y):
        losses = None
        t_list = []        
        y_sample_list = []
        epsilon_list = []
        for i in range(0, self.time_batch):
            t_list.append(self.sample_time(self.device))
            y_sample_temp, epsilon_temp = self.diff_Y.q_sample(y, t_list[-1].item())
            y_sample_list.append(y_sample_temp)
            epsilon_list.append(epsilon_temp)

        t_cat = torch.cat(t_list, dim=0).view(-1,1)
        t_cat = t_cat.expand(-1, x.shape[0]//self.time_batch)
        t_cat = t_cat.reshape(-1)
        q_Y_sample = torch.cat(y_sample_list, dim=0)
        epsilon_list = torch.cat(epsilon_list, dim=0)
        orig_shapes = y.shape[0]
        pred_y = model(x, q_Y_sample, adj, t_cat, self.num_timesteps)

        for e, t in enumerate(t_list):
            # calculate vlb
            vb_loss = self.diff_Y.compute_vb_loss(pred_y=pred_y[orig_shapes*e:orig_shapes*(e+1)], x_0=y, x_t=q_Y_sample[orig_shapes*e:orig_shapes*(e+1)], t=t)
            vb_loss = vb_loss * self.time_batch / 1000
            if losses == None:
                losses = vb_loss
            else:
                losses = losses + vb_loss

            if self.unweighted_MSE:
                coef = 1
            else:
                if t == 1:
                    coef = 0.5/self.diff_Y.alphas[1]
                else:
                    coef = 0.5*((self.diff_Y.betas[t]**2)/(self.diff_Y.posterior_variance[t]*self.diff_Y.alphas[t]*(1-self.diff_Y.alphas_cumprod[t-1])))
            if losses == None:
                losses = coef*torch.mean(torch.sum(((pred_y[orig_shapes*e:orig_shapes*(e+1)][:, :2]-epsilon_list[orig_shapes*e:orig_shapes*(e+1)])**2), dim = 1))
            else:
                losses = losses + coef*torch.mean(torch.sum(((pred_y[orig_shapes*e:orig_shapes*(e+1)][:, :2]-epsilon_list[orig_shapes*e:orig_shapes*(e+1)])**2), dim = 1))

        return losses/self.time_batch

    def sample_time(self, device):
        t = torch.randint(1, self.num_timesteps+1, (1,), device=device).long()
        return t

    def test(self, model, x, adj, y, data, noise_temp=0.001):
        updated_y = torch.randn_like(y)*noise_temp
        for i in range(0, self.diff_Y.num_timesteps):
            eps = model(x, updated_y, adj, torch.tensor([self.diff_Y.num_timesteps-i]).to(x.device).expand(x.shape[0]), self.diff_Y.num_timesteps)
            updated_y = (1/self.diff_Y.sqrt_alphas[self.diff_Y.num_timesteps-i])*(updated_y- (self.diff_Y.thresh[self.num_timesteps-i])*eps[:, :2])
            updated_y = updated_y + torch.sqrt(self.diff_Y.posterior_variance[self.diff_Y.num_timesteps-i])*torch.randn_like(eps[:, :2])*noise_temp

        return data, updated_y, y
