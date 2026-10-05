"""Per-image metrics, including valid soft semantic-region errors."""
import math
import torch
import torch.nn.functional as F
from .common import luminance


def ssim(output, target):
    """RGB SSIM with an 11x11 Gaussian window and replicated borders.

    Population covariance, unit data range, sigma=1.5. This same window is
    defined for tiny images; it is not skimage's reflected-border crop SSIM.
    """
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


def image_metrics(output, target, semantics=None, valid=None, confidence=None):
    if output.shape != target.shape or output.ndim != 4 or output.shape[0] != 1:
        raise ValueError('metrics require matching single-image BCHW tensors')
    mse = (output-target).square().mean().item()
    # Finite perfect-image convention keeps reports portable strict JSON.
    metrics = {'psnr': -10*math.log10(max(mse, 1e-12)), 'ssim':ssim(output,target),
               'rgb_l1':(output-target).abs().mean().item(),
               'luma_l1':(luminance(output)-luminance(target)).abs().mean().item(),
               'clipping_fraction':((output <= 0)|(output >= 1)).float().mean().item()}
    regions = {}
    if semantics is not None and valid is not None:
        confidence = torch.ones_like(semantics[:, :1]) if confidence is None else confidence
        weights = semantics * valid * confidence
        rgb_error = (output-target).abs().mean(1, keepdim=True)
        y_error = (luminance(output)-luminance(target)).abs()
        for channel in range(semantics.shape[1]):
            weight = weights[:, channel:channel+1]
            mass = weight.sum().item()
            if mass > 0:
                name = ('person','skin','sky')[channel] if channel < 3 else f'channel_{channel}'
                regions[name] = {'rgb_l1':(rgb_error*weight).sum().item()/mass,
                                 'luma_l1':(y_error*weight).sum().item()/mass,
                                 'soft_pixel_mass':mass}
    metrics['regions'] = regions
    return metrics


def aggregate(records):
    """Arithmetic image means; missing regions are excluded, never zero-filled."""
    def mean_rows(rows):
        result = {'count':len(rows)}
        if rows:
            for key in rows[0]['metrics']:
                if key != 'regions':
                    result[key] = sum(row['metrics'][key] for row in rows) / len(rows)
        return result
    report = {'global':mean_rows(records), 'by_scene':{}, 'by_camera':{}, 'by_region':{}}
    for group, field in [('by_scene','scene'), ('by_camera','camera')]:
        for label in sorted({row[field] for row in records}):
            report[group][label] = mean_rows([row for row in records if row[field] == label])
    labels = sorted({name for row in records for name in row['metrics']['regions']})
    for name in labels:
        values = [row['metrics']['regions'][name] for row in records if name in row['metrics']['regions']]
        report['by_region'][name] = {'count':len(values), **{key:sum(row[key] for row in values)/len(values) for key in values[0]}}
    return report
