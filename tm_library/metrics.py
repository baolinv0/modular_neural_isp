"""Display, linear-stage, and target-referenced regional color diagnostics.

Display tensors are normalized sRGB. Optional ``linear`` is the pre-chroma
operator stage, compared with decoded final target sRGB. That final target
includes downstream tone/color rendering: these linear_l1/log/gradient
diagnostics are not calibrated ground truth errors at the operator stage.
``preclip`` is an optional raw final
sRGB display stage before clipping, not the linear stage. Boundary-gradient
error measures disagreement with the target at valid semantic edges; it is
not a general halo detector. Hue is excluded when either color is achromatic.
"""
import math
import torch
import torch.nn.functional as F
from .common import luminance


def _srgb_to_linear(image):
    return torch.where(image <= .04045, image / 12.92,
                       ((image + .055) / 1.055).clamp_min(0).pow(2.4))


def srgb_to_lab(image):
    """sRGB BCHW -> CIE L*a*b*, D65 white, using the standard sRGB matrix."""
    linear = _srgb_to_linear(image)
    matrix = image.new_tensor([[.4124564,.3575761,.1804375],
                               [.2126729,.7151522,.0721750],
                               [.0193339,.1191920,.9503041]])
    xyz = torch.einsum('ij,bjhw->bihw', matrix, linear)
    xyz = xyz / image.new_tensor([.95047,1.,1.08883])[None,:,None,None]
    delta = 6/29
    f = torch.where(xyz > delta**3, xyz.clamp_min(0).pow(1/3),
                    xyz/(3*delta**2)+4/29)
    return torch.stack([116*f[:,1]-16, 500*(f[:,0]-f[:,1]),
                        200*(f[:,1]-f[:,2])], dim=1)


def ssim(output, target):
    """RGB SSIM: 11x11 Gaussian, sigma=1.5, replicated borders, range one."""
    coords = torch.arange(11, device=output.device, dtype=output.dtype) - 5
    kernel = torch.exp(-coords.square() / (2*1.5**2))
    kernel = kernel / kernel.sum()
    window = (kernel[:, None]*kernel[None, :])[None, None].expand(3, 1, 11, 11)
    def smooth(x):
        return F.conv2d(F.pad(x, (5,5,5,5), mode='replicate'), window, groups=3)
    mu_x, mu_y = smooth(output), smooth(target)
    var_x = (smooth(output.square())-mu_x.square()).clamp_min(0)
    var_y = (smooth(target.square())-mu_y.square()).clamp_min(0)
    cov = smooth(output*target)-mu_x*mu_y
    return (((2*mu_x*mu_y+.01**2)*(2*cov+.03**2)) /
            ((mu_x.square()+mu_y.square()+.01**2)*(var_x+var_y+.03**2))).mean().item()


def _validate_image(value, name, shape=None, normalized=False):
    if not isinstance(value, torch.Tensor) or value.ndim != 4 or value.shape[0] != 1 or value.shape[1] != 3 or min(value.shape[-2:]) < 1:
        raise ValueError(f'{name}: metrics require a single-image RGB BCHW tensor')
    if shape is not None and value.shape != shape:
        raise ValueError(f'{name}: metrics require matching single-image BCHW tensors')
    if not torch.isfinite(value).all():
        raise ValueError(f'{name}: pixels must be finite')
    if normalized and ((value < 0).any() or (value > 1).any()):
        raise ValueError(f'{name}: display pixels must be normalized [0,1] sRGB')


def _gradients(value):
    return value[...,1:]-value[...,:-1], value[...,1:,:]-value[...,:-1,:]


def _gradient_l1(output, target):
    pairs = list(zip(_gradients(output), _gradients(target)))
    count = sum(a.numel() for a,b in pairs)
    return sum((a-b).abs().sum().item() for a,b in pairs)/count if count else 0.


def _boundary_error(output, target, semantics, reliability):
    numerator = mass = 0.
    errors = [(a-b).abs() for a,b in zip(_gradients(luminance(output)), _gradients(luminance(target)))]
    for dimension, error in zip((-1,-2), errors):
        a = semantics.narrow(dimension,1,semantics.shape[dimension]-1)
        b = semantics.narrow(dimension,0,semantics.shape[dimension]-1)
        ra = reliability.narrow(dimension,1,reliability.shape[dimension]-1)
        rb = reliability.narrow(dimension,0,reliability.shape[dimension]-1)
        weight = (a-b).abs()*torch.minimum(ra,rb)
        mass += weight.sum().item()
        numerator += (weight*error).sum().item()
    return numerator/mass if mass > 0 else None


def image_metrics(output, target, semantics=None, valid=None, confidence=None, *, linear=None, preclip=None):
    """One image's metrics; optional diagnostic arguments preserve the old API.

    Lab/chroma are CIE units; hue is shortest angular error in degrees and is
    absent if no chromatic pixels exist (threshold C*=1e-3). Optional linear
    errors include unclipped values versus decoded final display target, not
    calibrated operator-stage ground truth: they do not claim HDR recovery.
    Semantic weights use independent probabilities times validity/confidence.
    """
    _validate_image(output, 'output', normalized=True)
    _validate_image(target, 'target', shape=output.shape, normalized=True)
    mse = (output-target).square().mean().item()
    rgb_error = (output-target).abs().mean(1, keepdim=True)
    y_error = (luminance(output)-luminance(target)).abs()
    lab_o, lab_t = srgb_to_lab(output), srgb_to_lab(target)
    delta_e = (lab_o-lab_t).square().sum(1,keepdim=True).sqrt()
    chroma_o = lab_o[:,1:].square().sum(1,keepdim=True).sqrt()
    chroma_t = lab_t[:,1:].square().sum(1,keepdim=True).sqrt()
    hue_valid = (chroma_o > 1e-3) & (chroma_t > 1e-3)
    hue_o = torch.atan2(lab_o[:,2:3],lab_o[:,1:2])
    hue_t = torch.atan2(lab_t[:,2:3],lab_t[:,1:2])
    difference = hue_o-hue_t
    hue_error = torch.atan2(difference.sin(),difference.cos()).abs()*(180/math.pi)
    metrics = {'psnr': -10*math.log10(max(mse, 1e-12)), 'ssim':ssim(output,target),
               'rgb_l1':rgb_error.mean().item(), 'luma_l1':y_error.mean().item(),
               'clipping_fraction':((output <= 0)|(output >= 1)).float().mean().item(),
               'delta_e76':delta_e.mean().item(),
               'chroma_l1':(chroma_o-chroma_t).abs().mean().item(),
               'hue_valid_fraction':hue_valid.float().mean().item()}
    if hue_valid.any():
        metrics['hue_degrees'] = hue_error[hue_valid].mean().item()
    if linear is not None:
        _validate_image(linear, 'linear', shape=output.shape)
        linear_target = _srgb_to_linear(target)
        metrics.update(linear_l1=(linear-linear_target).abs().mean().item(),
                       linear_log_l1=(torch.log1p(linear.clamp_min(0))-torch.log1p(linear_target)).abs().mean().item(),
                       linear_gradient_l1=_gradient_l1(linear,linear_target),
                       linear_below_zero_fraction=(linear<0).float().mean().item(),
                       linear_above_one_fraction=(linear>1).float().mean().item())
    if preclip is not None:
        _validate_image(preclip, 'preclip', shape=output.shape)
        metrics.update(preclip_below_zero_fraction=(preclip<0).float().mean().item(),
                       preclip_above_one_fraction=(preclip>1).float().mean().item(),
                       preclip_below_zero_excess=(-preclip).clamp_min(0).mean().item(),
                       preclip_above_one_excess=(preclip-1).clamp_min(0).mean().item())
    regions = {}
    if semantics is not None and valid is not None:
        if semantics.ndim != 4 or semantics.shape[0] != 1 or semantics.shape[1] < 1 or semantics.shape[-2:] != output.shape[-2:]:
            raise ValueError('semantics must have matching single-image spatial dimensions')
        confidence = torch.ones_like(semantics[:,:1]) if confidence is None else confidence
        for name,value in [('semantics',semantics),('valid',valid),('confidence',confidence)]:
            if value.ndim != 4 or value.shape[0] != 1 or value.shape[-2:] != output.shape[-2:] or value.shape[1] not in (1,semantics.shape[1]):
                raise ValueError(f'{name}: metric mask shape mismatch')
            if not torch.isfinite(value).all() or (value<0).any() or (value>1).any():
                raise ValueError(f'{name}: metric probabilities must be finite in [0,1]')
        reliability = (valid*confidence).expand_as(semantics)
        weights = semantics*reliability
        boundary = _boundary_error(output,target,semantics,reliability)
        if boundary is not None:
            metrics['semantic_boundary_luma_gradient_l1'] = boundary
        for channel in range(semantics.shape[1]):
            weight = weights[:,channel:channel+1]
            mass = weight.sum().item()
            if mass <= 0:
                continue
            def mean(value):
                return (value*weight).sum().item()/mass
            name = ('person','skin','sky')[channel] if channel < 3 else f'channel_{channel}'
            region = {'rgb_l1':mean(rgb_error), 'luma_l1':mean(y_error),
                      'delta_e76':mean(delta_e), 'chroma_l1':mean((chroma_o-chroma_t).abs()),
                      'output_lab_l':mean(lab_o[:,:1]), 'target_lab_l':mean(lab_t[:,:1]),
                      'output_chroma':mean(chroma_o), 'target_chroma':mean(chroma_t),
                      'soft_pixel_mass':mass}
            chromatic_weight = weight*hue_valid
            chromatic_mass = chromatic_weight.sum().item()
            region['hue_valid_fraction'] = chromatic_mass/mass
            if chromatic_mass > 0:
                region['hue_degrees'] = (hue_error*chromatic_weight).sum().item()/chromatic_mass
            regions[name] = region
    metrics['regions'] = regions
    return metrics


def aggregate(records, *, include_per_image=False, group_scenario=True):
    """Arithmetic image means, excluding absent optional values/empty regions."""
    def mean_metrics(values):
        result = {'count':len(values)}
        for key in sorted({key for value in values for key in value if key != 'regions'}):
            present = [value[key] for value in values if key in value]
            result[key] = sum(present)/len(present)
        return result
    def mean_rows(rows):
        return mean_metrics([row['metrics'] for row in rows])
    report = {'global':mean_rows(records), 'by_scene':{}, 'by_camera':{}, 'by_region':{}}
    groups = [('by_scene','scene'), ('by_camera','camera')]
    if group_scenario:
        report['by_scenario'] = {}
        groups.append(('by_scenario','scenario'))
    for group,field in groups:
        for label in sorted({row.get(field,'unknown') for row in records}):
            report[group][label] = mean_rows([row for row in records if row.get(field,'unknown') == label])
    labels = sorted({name for row in records for name in row['metrics'].get('regions',{})})
    for name in labels:
        values = [row['metrics']['regions'][name] for row in records if name in row['metrics'].get('regions',{})]
        report['by_region'][name] = mean_metrics(values)
    if include_per_image:
        report['per_image'] = list(records)
    return report
