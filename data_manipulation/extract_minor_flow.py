import math

import torch
import numpy as np


def compute_secondary_flows(target, num_iterations):
    """
    Computes a mixture model of two gaussians, to get hold of the primary and secondary flow
    directions.
    :param target:
    :param num_iterations:
    :return:
    """
    skip_n_levels = 2
    b, c, w, h = target.size()
    full_size = np.array([w, h], dtype=np.int)
    target_size = np.array([w, h], dtype=np.int)

    targets_primary = []
    targets_secondary = []

    for i in range(6):
        target_size = (target_size + 1) // 2
        if i < (skip_n_levels - 1):
            targets_primary.append(None)
            targets_secondary.append(None)
            continue

        tw = target_size[0]
        th = target_size[1]
        num_patches = tw*th

        patch_size = np.ceil(full_size / target_size).astype(int)
        diff = (target_size * patch_size) - full_size  # we need to add padding

        scale = 2
        patch_size //= scale

        # we scale the image up a bit, so that all patches are equal in size
        corrected_size = full_size + diff
        interpolated_target = torch.nn.functional.interpolate(target, [corrected_size[0] // scale, corrected_size[1] // scale], mode="bilinear", align_corners=True)
        #print("from ", full_size, interpolated_target.shape, target_size, "patch", patch_size)

        patches = interpolated_target\
            .unfold(2, int(patch_size[0]), int(patch_size[0]))\
            .unfold(3, int(patch_size[1]), int(patch_size[1]))
        pool_size = patch_size[0] * patch_size[1]
        patches = patches.contiguous().view(b, c, -1, pool_size)
        pool_size = patches.shape[3]

        # every patch now contains all values in the scope of a pool.
        # patch size: [batches, channels, pool_size, target_w * target_h]

        # approximate gaussians
        # we fit a gaussian mixture model

        # create gaussians for every pool
        # TODO initialize random
        mu_a = torch.zeros((b, 2, num_patches)).cuda()
        mu_a[:, 0, :] = -10
        mu_b = torch.zeros((b, 2, num_patches)).cuda()
        mu_b[:, 0, :] = +10
        mu_a_broad = mu_a.unsqueeze(3).expand(-1, -1, -1, pool_size)
        mu_b_broad = mu_b.unsqueeze(3).expand(-1, -1, -1, pool_size)

        var_a = torch.ones((b, 2, num_patches)).cuda() * 10
        var_b = torch.ones((b, 2, num_patches)).cuda() * 10

        prior_a = (torch.ones(b, 1, num_patches).cuda() * 0.5)
        prior_b = (torch.ones(b, 1, num_patches).cuda() * 0.5)

        for i in range(num_iterations):
            # E
            # notice: we model var as I*o^2 -> correlation equals zero

            var_a_broad = (var_a + 1e-10).unsqueeze(3).expand(-1, -1, -1, pool_size)
            var_b_broad = (var_b + 1e-10).unsqueeze(3).expand(-1, -1, -1, pool_size)

            prior_a_broad = prior_a.unsqueeze(3).expand(-1, -1, -1, pool_size)
            prior_b_broad = prior_b.unsqueeze(3).expand(-1, -1, -1, pool_size)

            def no_diag_gaus(prior, mu_broad, var_broad):
                # ignoring the normalizing factor, as the results get normalized afterwards ...
                prob = prior[:, 0, :, :] * \
                         torch.exp(
                             -0.5 *
                             (
                                     ((patches[:, 0, :, :] - mu_broad[:, 0, :, :]) ** 2 / var_broad[:, 0, :, :]) +
                                     ((patches[:, 1, :, :] - mu_broad[:, 1, :, :]) ** 2 / var_broad[:, 1, :, :])
                             )
                         )
                return prob.unsqueeze(1)

            prob_a = no_diag_gaus(prior_a_broad, mu_a_broad, var_a_broad) + 1e-10
            prob_b = no_diag_gaus(prior_b_broad, mu_b_broad, var_b_broad) + 1e-10

            prob_ab_sum = prob_a + prob_b
            #print("$>", torch.min(prob_ab_sum))
            prob_a /= prob_ab_sum
            prob_b /= prob_ab_sum

            # M
            prob_sum_a = torch.sum(prob_a, dim=3)
            prob_sum_b = torch.sum(prob_b, dim=3)
            #prob_sum_b = pool_size - prob_sum_a

            mu_a = (1 / prob_sum_a) * torch.sum(prob_a * patches, dim=3)
            mu_b = (1 / prob_sum_b) * torch.sum(prob_b * patches, dim=3)

            mu_a_broad = mu_a.unsqueeze(3).expand(-1, -1, -1, pool_size)
            mu_b_broad = mu_b.unsqueeze(3).expand(-1, -1, -1, pool_size)

            #print("!>>", patches.shape, mu_a.shape, mu_a_broad.shape)
            var_a = (1 / prob_sum_a) * torch.sum(prob_a * (patches - mu_a_broad)**2, dim=3)
            var_b = (1 / prob_sum_b) * torch.sum(prob_b * (patches - mu_b_broad)**2, dim=3)

            # compute class prior
            prior_a = torch.sum(prob_a, dim=3) / pool_size
            prior_b = 1 - prior_a
            #print("$$", prior_a, prior_b)

            #print("<>", prob_a.shape, prior_a.shape)

        a_is_primary = prior_a > prior_b
        a_is_primary = prior_a > prior_b
        #print("!!!", torch.sum(a_is_primary).cpu().numpy())
        #print("!!!", torch.sum(~a_is_primary).cpu().numpy())
        # the gaussian with more weight is the primary flow
        target_primary = torch.where(a_is_primary, mu_a, mu_b).view([b, 2, tw, th])
        target_secondary = torch.where(a_is_primary, mu_b, mu_a).view([b, 2, tw, th])

        #print("=>", prior_a.shape, mu_a.shape, target_primary.shape)

        targets_primary.append(target_primary)
        targets_secondary.append(target_secondary)

    return targets_primary[::-1], targets_secondary[::-1]