# GRU-VAE Refactor Blueprint

## Purpose
- Retain the VAE foundation while restructuring the project around self-supervised noise injection.
- Transition training to a noise-adding scheme on clean measurements, producing supervision masks aligned with the injected noise.
 - Deploy the trained model for FDIA localization and recovery without relying on any legacy fill-in logic.
 - Streamline the repository by removing legacy fill-in pathways and preserving only the minimal modules required for noise-based self-supervision and FDIA validation.

## Proposed Approach
1. **Codebase Audit**  
   Map every legacy fill-in related module, script, and configuration to plan safe removal or replacement. Identify shared utilities that must be retained for data handling, logging, and evaluation.
2. **Noise Injection Pipeline**  
   Define configurable noise policies (single-point and windowed corruption), implement mask generation, and integrate them into the data loaders for self-supervised training.
3. **Model Interface Revision**  
   Expose a clean training API that ingests noisy inputs and noise labels, ensuring loss functions and metrics align with the new supervision signal while preserving VAE core components.
4. **Training Orchestration**  
   Refresh training scripts and configuration files to reflect the new pipeline: data preparation, batching, optimizer schedules, checkpointing, and monitoring hooks.
5. **Inference for FDIA**  
   Implement a focused inference module that runs localization and reconstruction on suspected FDIA data, exporting artifacts needed for downstream evaluation.
6. **Validation Scripts**  
   Deliver two English-only verification entry points: single-point attack simulation and sliding-window attack scenarios, both reporting localization accuracy and reconstruction fidelity.
7. **Repository Cleanup**  
   Remove deprecated fill-in assets (code, configs, docs), update README references, and ensure remaining modules use UTF-8 encoding and follow naming conventions.

## Deliverables
- Self-supervised training scripts and configs tailored to noise augmentation.
- FDIA-focused inference and validation scripts covering single-point and sliding-window attacks.
- A lean project structure free from legacy fill-in code paths, documented for future contributors.
