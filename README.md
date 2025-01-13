Optical Music Recognition.

This uses the GrandPiano dataset to train a Transformer network to translate sheet music
aka images, into a simplified *kern representation. The simplification of *kern consists
mostly in droping any features not related playback per se, e.g. dropping ties, beams, etc.
In addition, all voices, or chords are merged into a single "spine".

The FILES files has infos about the role & purpose of every file in the package.

This is not meant for public consumption yet.
