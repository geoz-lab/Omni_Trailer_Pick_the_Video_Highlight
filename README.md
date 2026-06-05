# Omni Trailer — Pick the Video Highlight

Automatically identify and extract the most compelling moments from a video to
assemble a short highlight reel or trailer.

> **Note:** This README was generated as a starting point. Update the sections
> below to match the project's actual implementation, dependencies, and usage.

## Overview

Omni Trailer analyzes an input video, scores segments by how "highlight-worthy"
they are, and stitches the top moments together into a trailer-length clip.

## Features

- Scene / shot detection to split a video into candidate segments
- Highlight scoring of each segment
- Automatic assembly of top segments into a single output clip
- Configurable target length for the final trailer

## Installation

```bash
git clone https://github.com/geoz-lab/Omni_Trailer_Pick_the_Video_Highlight.git
cd Omni_Trailer_Pick_the_Video_Highlight
# install dependencies, e.g.:
# pip install -r requirements.txt
```

## Usage

```bash
# Example — adjust to the actual entry point:
# python omni_trailer.py --input input.mp4 --output trailer.mp4 --length 30
```

## License

This project is licensed under the [MIT License](LICENSE).
