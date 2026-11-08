# Sample Assets & Generated Results

This folder contains verified test assets and high-fidelity stylized results for Neural Style Transfer:

## Content Images (`assets/content/`)
- `dancing.jpg`: Classic ballerina photograph. Clear human anatomy, dynamic posture, and subtle ambient lighting.
- `bridge.jpg`: High-resolution architectural photography of the Golden Gate Bridge with distinct suspension cables and calm water.
- `neckarfront.png`: Synthetic color block image used for fast deterministic CPU unit and integration tests.

## Style Images (`assets/style/`)
- `picasso.jpg`: Pablo Picasso cubist portrait. Strong geometric lines, angular facets, and distinctive earth-tone palette.
- `monet.jpg`: Claude Monet impressionist *Water Lilies* oil painting. Dappled pastel reflections and prominent impasto texture.
- `starry_night.png`: Vincent van Gogh's *The Starry Night*. Iconic swirling sky motifs and cobalt blue brushwork.

## Verified Results / Proof Gallery (`assets/results/`)
- `picasso_dancer.png`: Synthesized cubist ballerina (100 L-BFGS steps, 4.2s on NVIDIA GeForce RTX 3050).
- `monet_bridge.png`: Synthesized impressionist Golden Gate Bridge (100 L-BFGS steps, 4.9s on NVIDIA GeForce RTX 3050).
