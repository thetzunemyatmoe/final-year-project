from torch import nn
import torch
import numpy as np
import torch.nn.functional as F


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)


def set_init(layers):
    for layer in layers:
        nn.init.normal_(layer.weight, mean=0., std=0.1)
        nn.init.constant_(layer.bias, 0.)


def push_and_pull(local_actor, local_critic, global_actor, global_critic, optimizer, done, s_, bs, ba, br, gamma):
    # Terminal state
    if done:
        v_s_ = 0.
    else:
        v_s_ = local_critic.forward(v_wrap(s_[None, :]))[-1].data.numpy()[0]

    buffer_v_target = []
    # Reverese iteraction
    for r in br[::-1]:
        v_s_ = r + gamma * v_s_
        buffer_v_target.append(v_s_)
    buffer_v_target.reverse()

    # Local loss
    loss = loss_function(local_actor=local_actor,
                         local_critic=local_critic,
                         state=v_wrap(np.vstack(bs)),
                         action=v_wrap(np.array(ba), dtype=np.int64),
                         value_t=v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients and push local parameters to global
    optimizer.zero_grad()
    loss.backward()
    for lcp, gcp in zip(local_critic.parameters(), global_critic.parameters()):
        gcp._grad = lcp.grad

    for lap, gap in zip(local_actor.parameters(), global_actor.parameters()):
        gap._grad = lap.grad
    optimizer.step()
    optimizer.step()

    # pull global parameters
    local_actor.load_state_dict(global_actor.state_dict())
    local_critic.load_state_dict(global_critic.state_dict())


def loss_function(local_actor, local_critic, state, action, value_t, beta=0.05):
    logits = local_actor.forward(state)
    values = local_critic.forward(state)

    # Critic loss
    temp_diff = value_t - values
    critic_loss = temp_diff.pow(2)

    # Actor loss
    probs = F.softmax(logits, dim=1)
    m = local_actor.distribution(probs=probs)
    log_prob = m.log_prob(action)
    entropy = m.entropy()

    actor_loss = - \
        (log_prob * temp_diff.detach().squeeze())
    total_loss = (critic_loss + actor_loss).mean()
    return total_loss


def record(global_ep, global_ep_r, ep_r, res_queue, name):
    with global_ep.get_lock():
        global_ep.value += 1
    with global_ep_r.get_lock():
        if global_ep_r.value == 0.:
            global_ep_r.value = ep_r
        else:
            global_ep_r.value = global_ep_r.value * 0.99 + ep_r * 0.01
    res_queue.put(global_ep_r.value)
    print(
        name,
        "Ep:", global_ep.value,
        "| Ep_r: %.0f" % global_ep_r.value,
    )
