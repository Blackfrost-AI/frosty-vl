# Distribution notice

## What this kit contains

This kit contains only Blackfrost's serving code and its bundled direction
vectors — together, "the plugin." It ships **no model weights of any kind.**

## How the plugin works

Frosty VL is a persona-steering plugin for Qwen3-VL text encoders. At runtime
it projects its direction vectors onto a copy, held in memory, of the base
model's Qwen3-VL text encoder (Apache-2.0, Qwen / Alibaba). The plugin never
modifies, ships, hosts, or redistributes any base-model weights. Files on disk
remain untouched.

## The base model is yours

You supply your own Qwen3-VL-based diffusion pipeline as the base model. You
obtain that base model, and you are responsible for complying with its license
and terms. Blackfrost does not distribute it and takes no position on your
right to use it.

## Rights in the plugin

The bundled direction vectors and the Blackfrost serving code are Blackfrost
intellectual property, delivered for use inside your licensed environment. Do
not redistribute them outside the licensed customer environment.

## Third-party base image

The NVIDIA base container image and its contents remain subject to NVIDIA's
applicable container and software license terms.
