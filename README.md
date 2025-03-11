# Optical Music Recognition.

This uses the GrandPiano dataset to train a Transformer network to translate sheet music
aka images, into a simplified *kern representation. The simplification of *kern consists
mostly in droping any features not related playback per se, e.g. dropping ties, beams, etc.
In addition, all voices, or chords are merged into a single "spine".

The FILES files has infos about the role & purpose of every file in the package.

This is not meant for public consumption yet.

Datasets:
- GrandStaff https://grfia.dlsi.ua.es/musicdocs/grandstaff.tgz
- Kern repository http://kern.ccarh.org/
- asap-dataset: https://github.com/fosfrancesco/asap-dataset

KernSheet dataset

The KernSheet dataset will eventually map bars within a score (aka pdf) 
into the corresponding bars within the kern file. Here is how to cook it:

```shell
# ./kernsheet.py make-kern-sheet TARGET
```
TARGET is the (existing) directory in which you want to create the 
dataset. This command will fetch a bunch of .zip files from KernScore
and copy them locally.

```shell
# ./kernsheet.py merge-asap ASAP TARGET
```
ASAP is a direxctory in which you've cloned the asap-dataset (see above)
and TARGET is your target KernSheet dataset directory. This command will
convert all .musicxml files into .krn file and move them to their appropriate
spot in the final dataset directory.

From there on, kernsheet.sh has a bunch of command to help you edit the 
dataset, this includes:
- finding one or more pdf score for a given kern file,
- reviewing and validating the page layout of each pdf,
- ensuring the mapping to the kern file is correct.


