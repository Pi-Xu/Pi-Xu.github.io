---
title: "Issue with Recording Videos in Mujoco Environment"
date: 2024-01-08T22:37:39+08:00
draft: false
author:
tags: [mujoco-py, rl-env]
image:
description:
toc: true
---


## Error with env.render()

I encountered issues while attempting to record videos of the robot's operation and found problems with rendering:
1. Rendering with no image (After some googling, it seemed like something was `unset`, rendering worked but the image was completely black...)
2. `# Failed to load OpenGL` (If I didn't `unset` that thing, this problem occurred)

I found a solution mentioned in an issue [here](https://github.com/openai/mujoco-py/issues/665#issuecomment-1049503083):

> There have been multiple issues which relate to the same error ([#598](https://github.com/openai/mujoco-py/issues/598), [#187](https://github.com/openai/mujoco-py/issues/187), [#390](https://github.com/openai/mujoco-py/issues/390)) but unfortunately none of them worked for me. What worked was to change [this line](https://github.com/openai/gym/blob/c8321e68bbd7e452ef65fc237525669979bb45af/gym/envs/mujoco/mujoco_env.py#L193) in `<site_packages>/gym/envs/mujoco/mujoco_env.py` to:
> ```python
> self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, None, -1)
> ```
> You can find the location of `<site_packages>` in your system by using `pip show gym`.

If using Gymnasium, find `gymnasium/envs/mujoco/mujoco_env.py` in `<site_packages>` and modify `self.viewer = mujoco_py.MjRenderContextOffscreen(self.sim, None, -1)` on this line.

Some version information:
```bash
gym                       0.26.2                   pypi_0    pypi
gym-notices               0.0.8                    pypi_0    pypi
gymnasium                 0.26.3                   pypi_0    pypi
gymnasium-notices         0.0.1                    pypi_0    pypi
mujoco                    3.1.1                    pypi_0    pypi
mujoco-py                 2.1.2.14                 pypi_0    pypi
```