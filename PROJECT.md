# spikelab

A workbench for developing spiking neural networks.

## Goal
An engine that simulates spiking networks, and an interface through which a person changes the neuron model, the connections, the architecture, the parameters and the learning rule without touching engine code. The learning rules include STDP and gradient training through the spike's discontinuity (surrogate gradients or a better answer you can defend). Then a simple document with diagrams that shows the owner what was built and how to use it.

## Why
Owner, #iiks1 1790020721.201009, 2026-09-21: "I want to make a spiking neural network with some engine and some interface that allows to change neuron models, connections, architecture, parameters for spiking neural network developemnt and changing learning roles including STDP and non continuous gradient descent or something. you are the expert in this, build it and write a simple document with diagrams to then show me".

## Boundaries
- It runs on this box. Run `box-status` before choosing a stack. No new hardware, no paid service.
- Neuron model, synapse, topology and learning rule are each swappable on their own, and a broken one does not take the others down.
- The interface is whatever lets him change things fastest from a laptop. A web page is served on localhost only. Opening it to the outside (a port, a tunnel, a hostname) is the owner's approval, asked once with a card.
- Correctness is shown, not claimed: known results reproduced (an LIF f-I curve, the STDP window, a small task learned under each rule).
- The document is short and reads like a person wrote it (style rule in `~/CLAUDE.md`). Diagrams are PNG or PDF files. No Artifacts.
- Spend is the box's first priority (`~/WORKING.md`): workers on Opus by default, subagents for bounded pieces, hand off at about 150k tokens.
- The repo is private under the owner's GitHub account. If making it is refused, ask in #iiks1-threads.

## Done
He changes each of those five things from the interface and sees the network behave differently. The document with its diagrams is posted as a file in this project's channel.

## Reporting
He steers in this project's channel. Open there with a short milestone plan. Tell `@main` in #iiks1 only at a milestone or a block.
