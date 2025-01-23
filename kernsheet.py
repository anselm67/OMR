#!/usr/bin/env python3
# To run this script, you'll need the following sources:
# KERN_SCORES_URL List of url to download zipped kern files from KernScore dataset
# A clone of https://github.com/fosfrancesco/asap-dataset
# Once you have these available, run the following commands:
# ./kernsheet.py make-kern-sheet TARGET_DIRECTORY
# ./kernsheet.py merge-asap ASAP_DIRECTORY TARGET_DIRECTORY
# At this point TARGET_DIRECTORY will contain the kernsheet dataset, with the
# following stats:
# file count : 689
# bad files  : 18
# bar count  : 72,808
# chord count: 565,965

import json
import logging
import os
import pickle
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast
from urllib.parse import quote

import click
import requests
from bs4 import BeautifulSoup
from cv2.typing import MatLike
from googlesearch import search

from imslp import IMSLP
from staffer import Staffer

KERN_SCORES_URL = {
    "classical": "https://kern.humdrum.org/cgi-bin/ksdata?l=/users/craig/classical&format=zip",
    "joplin": "https://kern.humdrum.org/cgi-bin/ksdata?l=users/craig/ragtime/joplin&format=zip",
    "inventions": "https://kern.humdrum.org/cgi-bin/ksdata?l=osu/classical/bach/inventions&format=zip"
}

PIANO_KERN_SCORES = [
    "bach/inventions/inven01.krn",
    "bach/inventions/inven02.krn",
    "bach/inventions/inven03.krn",
    "bach/inventions/inven04.krn",
    "bach/inventions/inven05.krn",
    "bach/inventions/inven06.krn",
    "bach/inventions/inven07.krn",
    "bach/inventions/inven08.krn",
    "bach/inventions/inven09.krn",
    "bach/inventions/inven10.krn",
    "bach/inventions/inven11.krn",
    "bach/inventions/inven12.krn",
    "bach/inventions/inven13.krn",
    "bach/inventions/inven14.krn",
    "bach/inventions/inven15.krn",
    "bach/inventions/sinfo01.krn",
    "bach/inventions/sinfo02.krn",
    "bach/inventions/sinfo03.krn",
    "bach/inventions/sinfo04.krn",
    "bach/inventions/sinfo05.krn",
    "bach/inventions/sinfo06.krn",
    "bach/inventions/sinfo07.krn",
    "bach/inventions/sinfo08.krn",
    "bach/inventions/sinfo09.krn",
    "bach/inventions/sinfo10.krn",
    "bach/inventions/sinfo11.krn",
    "bach/inventions/sinfo12.krn",
    "bach/inventions/sinfo13.krn",
    "bach/inventions/sinfo14.krn",
    "bach/inventions/sinfo15.krn",
    "bach/keyboard/bwv978-1.krn",
    "bach/keyboard/bwv978-2.krn",
    "bach/keyboard/bwv978-3.krn",
    "bach/wtc2preludes/wtc2p05.krn",
    "bach/wtc2preludes/wtc2p07.krn",
    "bach/wtc2preludes/wtc2p11.krn",
    "bach/wtc2preludes/wtc2p12.krn",
    "bach/wtc2preludes/wtc2p13.krn",
    "bach/wtc2preludes/wtc2p14.krn",
    "bach/wtc2preludes/wtc2p17.krn",
    "bach/wtc2preludes/wtc2p21.krn",
    "beethoven/piano/misc/rondo-op129.krn",
    "beethoven/piano/reduction/symph5-1.krn",
    "beethoven/piano/sonata/sonata01-1.krn",
    "beethoven/piano/sonata/sonata01-2.krn",
    "beethoven/piano/sonata/sonata01-3.krn",
    "beethoven/piano/sonata/sonata01-4.krn",
    "beethoven/piano/sonata/sonata02-1.krn",
    "beethoven/piano/sonata/sonata02-2.krn",
    "beethoven/piano/sonata/sonata02-3.krn",
    "beethoven/piano/sonata/sonata02-4.krn",
    "beethoven/piano/sonata/sonata03-1.krn",
    "beethoven/piano/sonata/sonata03-2.krn",
    "beethoven/piano/sonata/sonata03-3.krn",
    "beethoven/piano/sonata/sonata03-4.krn",
    "beethoven/piano/sonata/sonata04-1.krn",
    "beethoven/piano/sonata/sonata04-2.krn",
    "beethoven/piano/sonata/sonata04-3.krn",
    "beethoven/piano/sonata/sonata04-4.krn",
    "beethoven/piano/sonata/sonata05-1.krn",
    "beethoven/piano/sonata/sonata05-2.krn",
    "beethoven/piano/sonata/sonata05-3.krn",
    "beethoven/piano/sonata/sonata06-1.krn",
    "beethoven/piano/sonata/sonata06-2.krn",
    "beethoven/piano/sonata/sonata06-3.krn",
    "beethoven/piano/sonata/sonata07-1.krn",
    "beethoven/piano/sonata/sonata07-2.krn",
    "beethoven/piano/sonata/sonata07-3.krn",
    "beethoven/piano/sonata/sonata07-4.krn",
    "beethoven/piano/sonata/sonata08-1.krn",
    "beethoven/piano/sonata/sonata08-2.krn",
    "beethoven/piano/sonata/sonata08-3.krn",
    "beethoven/piano/sonata/sonata09-1.krn",
    "beethoven/piano/sonata/sonata09-2.krn",
    "beethoven/piano/sonata/sonata09-3.krn",
    "beethoven/piano/sonata/sonata10-1.krn",
    "beethoven/piano/sonata/sonata10-2.krn",
    "beethoven/piano/sonata/sonata10-3.krn",
    "beethoven/piano/sonata/sonata11-1.krn",
    "beethoven/piano/sonata/sonata11-2.krn",
    "beethoven/piano/sonata/sonata11-3.krn",
    "beethoven/piano/sonata/sonata11-4.krn",
    "beethoven/piano/sonata/sonata12-1.krn",
    "beethoven/piano/sonata/sonata12-2.krn",
    "beethoven/piano/sonata/sonata12-3.krn",
    "beethoven/piano/sonata/sonata12-4.krn",
    "beethoven/piano/sonata/sonata13-1.krn",
    "beethoven/piano/sonata/sonata13-2.krn",
    "beethoven/piano/sonata/sonata13-3.krn",
    "beethoven/piano/sonata/sonata13-4.krn",
    "beethoven/piano/sonata/sonata14-1.krn",
    "beethoven/piano/sonata/sonata14-2.krn",
    "beethoven/piano/sonata/sonata14-3.krn",
    "beethoven/piano/sonata/sonata15-1.krn",
    "beethoven/piano/sonata/sonata15-2.krn",
    "beethoven/piano/sonata/sonata15-3.krn",
    "beethoven/piano/sonata/sonata15-4.krn",
    "beethoven/piano/sonata/sonata16-1.krn",
    "beethoven/piano/sonata/sonata16-2.krn",
    "beethoven/piano/sonata/sonata16-3.krn",
    "beethoven/piano/sonata/sonata17-1.krn",
    "beethoven/piano/sonata/sonata17-2.krn",
    "beethoven/piano/sonata/sonata17-3.krn",
    "beethoven/piano/sonata/sonata18-1.krn",
    "beethoven/piano/sonata/sonata18-2.krn",
    "beethoven/piano/sonata/sonata18-3.krn",
    "beethoven/piano/sonata/sonata18-4.krn",
    "beethoven/piano/sonata/sonata19-1.krn",
    "beethoven/piano/sonata/sonata19-2.krn",
    "beethoven/piano/sonata/sonata20-1.krn",
    "beethoven/piano/sonata/sonata20-2.krn",
    "beethoven/piano/sonata/sonata21-1.krn",
    "beethoven/piano/sonata/sonata21-2.krn",
    "beethoven/piano/sonata/sonata21-3.krn",
    "beethoven/piano/sonata/sonata21-4.krn",
    "beethoven/piano/sonata/sonata22-1.krn",
    "beethoven/piano/sonata/sonata22-2.krn",
    "beethoven/piano/sonata/sonata23-1.krn",
    "beethoven/piano/sonata/sonata23-2.krn",
    "beethoven/piano/sonata/sonata23-3.krn",
    "beethoven/piano/sonata/sonata24-1.krn",
    "beethoven/piano/sonata/sonata24-2.krn",
    "beethoven/piano/sonata/sonata25-1.krn",
    "beethoven/piano/sonata/sonata25-2.krn",
    "beethoven/piano/sonata/sonata25-3.krn",
    "beethoven/piano/sonata/sonata26-1.krn",
    "beethoven/piano/sonata/sonata26-2.krn",
    "beethoven/piano/sonata/sonata26-3.krn",
    "beethoven/piano/sonata/sonata27-1.krn",
    "beethoven/piano/sonata/sonata27-2.krn",
    "beethoven/piano/sonata/sonata28-1.krn",
    "beethoven/piano/sonata/sonata28-2.krn",
    "beethoven/piano/sonata/sonata28-3.krn",
    "beethoven/piano/sonata/sonata28-4.krn",
    "beethoven/piano/sonata/sonata29-1.krn",
    "beethoven/piano/sonata/sonata29-2.krn",
    "beethoven/piano/sonata/sonata29-3.krn",
    "beethoven/piano/sonata/sonata29-4.krn",
    "beethoven/piano/sonata/sonata30-1.krn",
    "beethoven/piano/sonata/sonata30-2.krn",
    "beethoven/piano/sonata/sonata30-3.krn",
    "beethoven/piano/sonata/sonata31-1.krn",
    "beethoven/piano/sonata/sonata31-2.krn",
    "beethoven/piano/sonata/sonata31-3.krn",
    "beethoven/piano/sonata/sonata32-1.krn",
    "beethoven/piano/sonata/sonata32-2.krn",
    "beethoven/piano/variations/swiss-0.krn",
    "beethoven/piano/variations/swiss-1.krn",
    "beethoven/piano/variations/swiss-2.krn",
    "beethoven/piano/variations/swiss-3.krn",
    "beethoven/piano/variations/swiss-4.krn",
    "beethoven/piano/variations/swiss-5.krn",
    "beethoven/piano/variations/swiss-6.krn",
    "brahms/op01/op01-2.krn",
    "brahms/op04/scherzo.krn",
    "brahms/op10/ballad10-1.krn",
    "chopin/ballade/ballade52.krn",
    "chopin/etude/etude10-02.krn",
    "chopin/etude/etude10-02b.krn",
    "chopin/etude/etude10-04.krn",
    "chopin/etude/etude10-05.krn",
    "chopin/etude/etude10-09.krn",
    "chopin/mazurka/mazurka-50.krn",
    "chopin/mazurka/mazurka-51.krn",
    "chopin/mazurka/mazurka-52.krn",
    "chopin/mazurka/mazurka06-1.krn",
    "chopin/mazurka/mazurka06-2.krn",
    "chopin/mazurka/mazurka06-3.krn",
    "chopin/mazurka/mazurka06-4.krn",
    "chopin/mazurka/mazurka07-1.krn",
    "chopin/mazurka/mazurka07-2.krn",
    "chopin/mazurka/mazurka07-3.krn",
    "chopin/mazurka/mazurka07-4.krn",
    "chopin/mazurka/mazurka07-5.krn",
    "chopin/mazurka/mazurka17-1.krn",
    "chopin/mazurka/mazurka17-2.krn",
    "chopin/mazurka/mazurka17-3.krn",
    "chopin/mazurka/mazurka17-4.krn",
    "chopin/mazurka/mazurka24-1.krn",
    "chopin/mazurka/mazurka24-2.krn",
    "chopin/mazurka/mazurka24-3.krn",
    "chopin/mazurka/mazurka24-4.krn",
    "chopin/mazurka/mazurka30-1.krn",
    "chopin/mazurka/mazurka30-2.krn",
    "chopin/mazurka/mazurka30-3.krn",
    "chopin/mazurka/mazurka30-4.krn",
    "chopin/mazurka/mazurka33-1.krn",
    "chopin/mazurka/mazurka33-2.krn",
    "chopin/mazurka/mazurka33-3.krn",
    "chopin/mazurka/mazurka33-4.krn",
    "chopin/mazurka/mazurka41-1.krn",
    "chopin/mazurka/mazurka41-2.krn",
    "chopin/mazurka/mazurka41-3.krn",
    "chopin/mazurka/mazurka41-4.krn",
    "chopin/mazurka/mazurka50-1.krn",
    "chopin/mazurka/mazurka50-2.krn",
    "chopin/mazurka/mazurka50-3.krn",
    "chopin/mazurka/mazurka56-1.krn",
    "chopin/mazurka/mazurka56-2.krn",
    "chopin/mazurka/mazurka56-3.krn",
    "chopin/mazurka/mazurka59-1.krn",
    "chopin/mazurka/mazurka59-2.krn",
    "chopin/mazurka/mazurka59-3.krn",
    "chopin/mazurka/mazurka63-1.krn",
    "chopin/mazurka/mazurka63-2.krn",
    "chopin/mazurka/mazurka63-3.krn",
    "chopin/mazurka/mazurka67-1.krn",
    "chopin/mazurka/mazurka67-2.krn",
    "chopin/mazurka/mazurka67-3.krn",
    "chopin/mazurka/mazurka67-4.krn",
    "chopin/mazurka/mazurka68-1.krn",
    "chopin/mazurka/mazurka68-2.krn",
    "chopin/mazurka/mazurka68-3.krn",
    "chopin/mazurka/mazurka68-4.krn",
    "chopin/nocturne/nocturne72-1.krn",
    "chopin/prelude/prelude28-01.krn",
    "chopin/prelude/prelude28-02.krn",
    "chopin/prelude/prelude28-03.krn",
    "chopin/prelude/prelude28-04.krn",
    "chopin/prelude/prelude28-05.krn",
    "chopin/prelude/prelude28-06.krn",
    "chopin/prelude/prelude28-07.krn",
    "chopin/prelude/prelude28-08.krn",
    "chopin/prelude/prelude28-09.krn",
    "chopin/prelude/prelude28-10.krn",
    "chopin/prelude/prelude28-11.krn",
    "chopin/prelude/prelude28-12.krn",
    "chopin/prelude/prelude28-13.krn",
    "chopin/prelude/prelude28-14.krn",
    "chopin/prelude/prelude28-15.krn",
    "chopin/prelude/prelude28-16.krn",
    "chopin/prelude/prelude28-17.krn",
    "chopin/prelude/prelude28-18.krn",
    "chopin/prelude/prelude28-19.krn",
    "chopin/prelude/prelude28-20.krn",
    "chopin/prelude/prelude28-21.krn",
    "chopin/prelude/prelude28-22.krn",
    "chopin/prelude/prelude28-23.krn",
    "chopin/prelude/prelude28-24.krn",
    "chopin/scherzo/scherzo2.krn",
    "chopin/waltz/waltz150.krn",
    "chopin/waltz/waltz64-1.krn",
    "chopin/waltz/waltz64-2.krn",
    "chopin/waltz/waltz69-2.krn",
    "clementi/op36/sonatina-36-1-1.krn",
    "clementi/op36/sonatina-36-1-2.krn",
    "clementi/op36/sonatina-36-1-3.krn",
    "clementi/op36/sonatina-36-2-1.krn",
    "clementi/op36/sonatina-36-2-2.krn",
    "clementi/op36/sonatina-36-2-3.krn",
    "clementi/op36/sonatina-36-3-1.krn",
    "clementi/op36/sonatina-36-3-2.krn",
    "clementi/op36/sonatina-36-3-3.krn",
    "clementi/op36/sonatina-36-4-1.krn",
    "clementi/op36/sonatina-36-4-2.krn",
    "clementi/op36/sonatina-36-4-3.krn",
    "clementi/op36/sonatina-36-5-1.krn",
    "clementi/op36/sonatina-36-5-2.krn",
    "clementi/op36/sonatina-36-5-3.krn",
    "clementi/op36/sonatina-36-6-1.krn",
    "clementi/op36/sonatina-36-6-2.krn",
    "field/nocturne.krn",
    "grieg/op01/op01-3.krn",
    "grieg/op03/op03-4.krn",
    "grieg/op06/op06-3.krn",
    "grieg/op07/op07-1.krn",
    "grieg/op12/op12-2.krn",
    "grieg/op17/op17-01.krn",
    "grieg/op43/butterfly.krn",
    "grieg/op43/erotic-poem.krn",
    "grieg/op43/little-bird.krn",
    "grieg/op43/native-country.krn",
    "grieg/op43/solitary-traveller.krn",
    "grieg/op43/to-spring.krn",
    "grieg/op66/op66-06.krn",
    "haydn/keyboard/uesonatas/sonata12-1.krn",
    "haydn/keyboard/uesonatas/sonata12-2.krn",
    "haydn/keyboard/uesonatas/sonata12-3.krn",
    "haydn/keyboard/uesonatas/sonata13-1.krn",
    "haydn/keyboard/uesonatas/sonata15-1.krn",
    "haydn/keyboard/uesonatas/sonata16-1.krn",
    "haydn/keyboard/uesonatas/sonata16-2.krn",
    "haydn/keyboard/uesonatas/sonata16-3.krn",
    "haydn/keyboard/uesonatas/sonata29-3.krn",
    "haydn/keyboard/uesonatas/sonata33-3.krn",
    "haydn/keyboard/uesonatas/sonata34-1.krn",
    "haydn/keyboard/uesonatas/sonata37-1.krn",
    "haydn/keyboard/uesonatas/sonata42-3.krn",
    "haydn/keyboard/uesonatas/sonata45-3a.krn",
    "haydn/keyboard/uesonatas/sonata49-1.krn",
    "haydn/keyboard/uesonatas/sonata50-1.krn",
    "haydn/keyboard/uesonatas/sonata51-3.krn",
    "haydn/keyboard/uesonatas/sonata52-3.krn",
    "haydn/keyboard/uesonatas/sonata53-3.krn",
    "haydn/keyboard/uesonatas/sonata59-1.krn",
    "haydn/keyboard/uesonatas/sonata61-1.krn",
    "haydn/keyboard/uesonatas/sonata61-2.krn",
    "haydn/keyboard/uesonatas/sonata62-1.krn",
    "haydn/keyboard/uesonatas/sonata62-2.krn",
    "haydn/keyboard/uesonatas/sonata62-3.krn",
    "hummel/op67/prelude67-01.krn",
    "hummel/op67/prelude67-02.krn",
    "hummel/op67/prelude67-03.krn",
    "hummel/op67/prelude67-04.krn",
    "hummel/op67/prelude67-05.krn",
    "hummel/op67/prelude67-06.krn",
    "hummel/op67/prelude67-07.krn",
    "hummel/op67/prelude67-08.krn",
    "hummel/op67/prelude67-09.krn",
    "hummel/op67/prelude67-10.krn",
    "hummel/op67/prelude67-11.krn",
    "hummel/op67/prelude67-12.krn",
    "hummel/op67/prelude67-13.krn",
    "hummel/op67/prelude67-14.krn",
    "hummel/op67/prelude67-15.krn",
    "hummel/op67/prelude67-16.krn",
    "hummel/op67/prelude67-17.krn",
    "hummel/op67/prelude67-18.krn",
    "hummel/op67/prelude67-19.krn",
    "hummel/op67/prelude67-20.krn",
    "hummel/op67/prelude67-21.krn",
    "hummel/op67/prelude67-22.krn",
    "hummel/op67/prelude67-23.krn",
    "hummel/op67/prelude67-24.krn",
    "joplin/antoinette.krn",
    "joplin/augustan.krn",
    "joplin/bethena.krn",
    "joplin/binks.krn",
    "joplin/breeze.krn",
    "joplin/cascades.krn",
    "joplin/chrysanthemum.krn",
    "joplin/cleopha.krn",
    "joplin/combination.krn",
    "joplin/countryclub.krn",
    "joplin/crush.krn",
    "joplin/easywinners.krn",
    "joplin/elite.krn",
    "joplin/entertainer.krn",
    "joplin/eugenia.krn",
    "joplin/favorite.krn",
    "joplin/felicity.krn",
    "joplin/figleaf.krn",
    "joplin/gladiolus.krn",
    "joplin/harmony.krn",
    "joplin/heliotrope.krn",
    "joplin/leola.krn",
    "joplin/lilyqueen.krn",
    "joplin/magnetic.krn",
    "joplin/majestic.krn",
    "joplin/mapleleaf.krn",
    "joplin/newrag.krn",
    "joplin/nonpareil.krn",
    "joplin/original.krn",
    "joplin/palmleaf.krn",
    "joplin/paragon.krn",
    "joplin/peacherine.krn",
    "joplin/pineapple.krn",
    "joplin/pleasant.krn",
    "joplin/reflection.krn",
    "joplin/rosebud.krn",
    "joplin/roseleaf.krn",
    "joplin/school.krn",
    "joplin/searchlight.krn",
    "joplin/solace.krn",
    "joplin/something.krn",
    "joplin/stoptime.krn",
    "joplin/sugarcane.krn",
    "joplin/sunflower.krn",
    "joplin/swipesy.krn",
    "joplin/wallstreet.krn",
    "joplin/weepingwillow.krn",
    "liszt/ballade2.krn",
    "macdowell/op01/op01.krn",
    "macdowell/op14/op14-5.krn",
    "macdowell/op39/op39-01.krn",
    "macdowell/op39/op39-02.krn",
    "macdowell/op39/op39-03.krn",
    "macdowell/op39/op39-04.krn",
    "macdowell/op39/op39-05.krn",
    "macdowell/op39/op39-06.krn",
    "mendelssohn/opus19-6.krn",
    "mendelssohn/opus62-3.krn",
    "mendelssohn/sonata1823.krn",
    "mozart/piano/sonata/sonata01-1.krn",
    "mozart/piano/sonata/sonata01-2.krn",
    "mozart/piano/sonata/sonata01-3.krn",
    "mozart/piano/sonata/sonata02-1.krn",
    "mozart/piano/sonata/sonata02-2.krn",
    "mozart/piano/sonata/sonata02-3.krn",
    "mozart/piano/sonata/sonata03-1.krn",
    "mozart/piano/sonata/sonata03-2.krn",
    "mozart/piano/sonata/sonata03-3.krn",
    "mozart/piano/sonata/sonata04-1.krn",
    "mozart/piano/sonata/sonata04-2.krn",
    "mozart/piano/sonata/sonata04-3.krn",
    "mozart/piano/sonata/sonata05-1.krn",
    "mozart/piano/sonata/sonata05-2.krn",
    "mozart/piano/sonata/sonata05-3.krn",
    "mozart/piano/sonata/sonata06-1.krn",
    "mozart/piano/sonata/sonata06-2.krn",
    "mozart/piano/sonata/sonata06-3a.krn",
    "mozart/piano/sonata/sonata06-3b.krn",
    "mozart/piano/sonata/sonata06-3c.krn",
    "mozart/piano/sonata/sonata06-3d.krn",
    "mozart/piano/sonata/sonata06-3e.krn",
    "mozart/piano/sonata/sonata06-3f.krn",
    "mozart/piano/sonata/sonata06-3g.krn",
    "mozart/piano/sonata/sonata06-3h.krn",
    "mozart/piano/sonata/sonata06-3i.krn",
    "mozart/piano/sonata/sonata06-3j.krn",
    "mozart/piano/sonata/sonata06-3k.krn",
    "mozart/piano/sonata/sonata06-3l.krn",
    "mozart/piano/sonata/sonata06-3m.krn",
    "mozart/piano/sonata/sonata07-1.krn",
    "mozart/piano/sonata/sonata07-2.krn",
    "mozart/piano/sonata/sonata07-3.krn",
    "mozart/piano/sonata/sonata08-1.krn",
    "mozart/piano/sonata/sonata08-2.krn",
    "mozart/piano/sonata/sonata08-3.krn",
    "mozart/piano/sonata/sonata09-1.krn",
    "mozart/piano/sonata/sonata09-2.krn",
    "mozart/piano/sonata/sonata09-3.krn",
    "mozart/piano/sonata/sonata10-1.krn",
    "mozart/piano/sonata/sonata10-2.krn",
    "mozart/piano/sonata/sonata10-3.krn",
    "mozart/piano/sonata/sonata11-1a.krn",
    "mozart/piano/sonata/sonata11-1b.krn",
    "mozart/piano/sonata/sonata11-1c.krn",
    "mozart/piano/sonata/sonata11-1d.krn",
    "mozart/piano/sonata/sonata11-1e.krn",
    "mozart/piano/sonata/sonata11-1f.krn",
    "mozart/piano/sonata/sonata11-1g.krn",
    "mozart/piano/sonata/sonata11-2.krn",
    "mozart/piano/sonata/sonata11-3.krn",
    "mozart/piano/sonata/sonata12-1.krn",
    "mozart/piano/sonata/sonata12-2.krn",
    "mozart/piano/sonata/sonata12-3.krn",
    "mozart/piano/sonata/sonata13-1.krn",
    "mozart/piano/sonata/sonata13-2.krn",
    "mozart/piano/sonata/sonata13-3.krn",
    "mozart/piano/sonata/sonata14-1.krn",
    "mozart/piano/sonata/sonata14-2.krn",
    "mozart/piano/sonata/sonata14-3.krn",
    "mozart/piano/sonata/sonata15-1.krn",
    "mozart/piano/sonata/sonata15-2.krn",
    "mozart/piano/sonata/sonata15-3.krn",
    "mozart/piano/sonata/sonata16-1.krn",
    "mozart/piano/sonata/sonata16-2.krn",
    "mozart/piano/sonata/sonata16-3.krn",
    "mozart/piano/sonata/sonata17-1.krn",
    "mozart/piano/sonata/sonata17-2.krn",
    "mozart/piano/sonata/sonata17-3.krn",
    "mozart/piano/sonatina/k439b.krn",
    "mozart/piano/sonatina/viennese1-1.krn",
    "mozart/piano/sonatina/viennese1-2.krn",
    "mozart/piano/sonatina/viennese1-3.krn",
    "mozart/piano/sonatina/viennese1-4.krn",
    "mozart/piano/sonatina/viennese6-1.krn",
    "mozart/piano/sonatina/viennese6-2.krn",
    "mozart/piano/sonatina/viennese6-3.krn",
    "mozart/piano/sonatina/viennese6-4.krn",
    "mozart/piano/variations/k265/k265-00.krn",
    "mozart/piano/variations/k265/k265-01.krn",
    "mussorgsky/exhibition/promenade.krn",
    "prokofiev/op22/visions22-1.krn",
    "prokofiev/op22/visions22-2.krn",
    "prokofiev/op22/visions22-3.krn",
    "ravel/sonatine-1.krn",
    "scarlatti/longo/L001K514.krn",
    "scarlatti/longo/L002K384.krn",
    "scarlatti/longo/L003K502.krn",
    "scarlatti/longo/L004K158.krn",
    "scarlatti/longo/L005K406.krn",
    "scarlatti/longo/L006K139.krn",
    "scarlatti/longo/L008K461.krn",
    "scarlatti/longo/L009K303.krn",
    "scarlatti/longo/L010K084.krn",
    "scarlatti/longo/L011K534.krn",
    "scarlatti/longo/L012K478.krn",
    "scarlatti/longo/L013K060.krn",
    "scarlatti/longo/L014K492.krn",
    "scarlatti/longo/L015K160.krn",
    "scarlatti/longo/L016K306.krn",
    "scarlatti/longo/L027K238.krn",
    "scarlatti/longo/L051K166.krn",
    "scarlatti/longo/L052K165.krn",
    "scarlatti/longo/L053K075.krn",
    "scarlatti/longo/L054K200.krn",
    "scarlatti/longo/L055K330.krn",
    "scarlatti/longo/L056K281.krn",
    "scarlatti/longo/L064K148.krn",
    "scarlatti/longo/L101K156.krn",
    "scarlatti/longo/L127K348.krn",
    "scarlatti/longo/L154K235.krn",
    "scarlatti/longo/L164K491.krn",
    "scarlatti/longo/L166K085.krn",
    "scarlatti/longo/L178K258.krn",
    "scarlatti/longo/L188K525.krn",
    "scarlatti/longo/L198K296.krn",
    "scarlatti/longo/L240K369.krn",
    "scarlatti/longo/L267K052.krn",
    "scarlatti/longo/L301K049.krn",
    "scarlatti/longo/L302K372.krn",
    "scarlatti/longo/L303K170.krn",
    "scarlatti/longo/L304K470.krn",
    "scarlatti/longo/L305K251.krn",
    "scarlatti/longo/L306K345.krn",
    "scarlatti/longo/L307K269.krn",
    "scarlatti/longo/L319K442.krn",
    "scarlatti/longo/L333K425.krn",
    "scarlatti/longo/L334K122.krn",
    "scarlatti/longo/L335K055.krn",
    "scarlatti/longo/L336K093.krn",
    "scarlatti/longo/L337K336.krn",
    "scarlatti/longo/L338K450.krn",
    "scarlatti/longo/L339K512.krn",
    "scarlatti/longo/L340K476.krn",
    "scarlatti/longo/L341K320.krn",
    "scarlatti/longo/L342K220.krn",
    "scarlatti/longo/L343K434.krn",
    "scarlatti/longo/L344K114.krn",
    "scarlatti/longo/L345K113.krn",
    "scarlatti/longo/L346K408.krn",
    "scarlatti/longo/L347K227.krn",
    "scarlatti/longo/L348K244.krn",
    "scarlatti/longo/L349K146.krn",
    "scarlatti/longo/L350K498.krn",
    "scarlatti/longo/L351K225.krn",
    "scarlatti/longo/L366K001.krn",
    "scarlatti/longo/L400K360.krn",
    "scarlatti/longo/L481K025.krn",
    "scarlatti/longo/L503K513.krn",
    "scarlatti/longo/L523K205.krn",
    "scriabin/op02/scriabin-op02_no01.krn",
    "scriabin/op08/scriabin-op08_no01.krn",
    "scriabin/op08/scriabin-op08_no02.krn",
    "scriabin/op08/scriabin-op08_no03.krn",
    "scriabin/op08/scriabin-op08_no04.krn",
    "scriabin/op08/scriabin-op08_no05.krn",
    "scriabin/op08/scriabin-op08_no06.krn",
    "scriabin/op11/scriabin-op11_no04.krn",
    "scriabin/op11/scriabin-op11_no15.krn",
    "scriabin/op59/prelude59-2.krn",
    "scriabin/op65/scriabin-op65_no02.krn",
    "scriabin/op65/scriabin-op65_no03.krn",
    "webern/webern-op27n1.krn",
    "webern/webern-op27n2.krn",
    "webern/webern-op27n3.krn",
]


def path_substract(shorter: Path, longer: Path) -> Path:
    prefix = os.path.commonprefix([shorter, longer])
    assert prefix is not None, f"Can't substract {shorter} from {longer}"
    return Path(os.path.relpath(longer, prefix))


def fetch(url: str, into: Path, refresh: bool = False):
    if into.exists() and not refresh:
        return
    print(f"Fetching {into.name}: from {url}")
    subprocess.run(["wget", "--quiet", "-O", str(into), url], check=True)


def unzip(target_directory: Path, zipfile: Path):
    subprocess.run(["unzip", "-o", "-d", str(target_directory), str(zipfile)])


def shell(commands: str):
    subprocess.run(commands, shell=True, check=False)


def verovio(file: Path, refresh: bool = False) -> bool:
    VEROVIO_HOME = "/home/anselm/Downloads/verovio"
    assert file.suffix == ".musicxml", "Expected suffix .musicxml"
    if file.with_suffix(".krn").exists() and not refresh:
        return False
    print(f"Translating {str(file)} to .krn")
    subprocess.run([
        f"{VEROVIO_HOME}/tools/verovio",
        "-r", f"{VEROVIO_HOME}/data",
        "-l", "off",
        "-f", "musicxml-hum", "-t", "hum",
        str(file),
        "-o", str(file.with_suffix(".krn"))
    ])
    return True


def is_likely_pdf(path: Path):
    """
    Checks if a file is likely a PDF by examining the first few bytes.

    Args:
    file_path: The path to the file.

    Returns:
    True if the file likely starts with a PDF header, False otherwise.
    """
    try:
        with open(path, 'rb') as f:
            header = f.read(5).decode('ascii')
            return header.startswith('%PDF-')
    except Exception as e:
        return False


@click.command()
@click.argument("target_directory", type=click.Path(dir_okay=True, exists=False),
                required=False,
                default="/home/anselm/datasets/kern-sheet")
def make_kern_sheet(target_directory: Path):
    target_directory = Path(target_directory)
    target_directory.mkdir(exist_ok=True)
    zip_directory = target_directory / "zip_sources"
    zip_directory.mkdir(exist_ok=True)
    # Fetches the urls and unzip the files, move things around.
    for name, url in KERN_SCORES_URL.items():
        zip_path = zip_directory / Path(name).with_suffix(".zip")
        fetch(url, zip_path)
        unzip(target_directory, zip_path)
    shell(
        f"cd {target_directory}\n"
        "mv users/craig/classical/* .\n"
        "mv inventions bach/\n"
        "find . -name CKSUM -exec rm {} \\; \n"
    )
    # Now remove all excess content.
    for root, _, filenames in os.walk(target_directory):
        for filename in filenames:
            file = Path(root) / filename
            rel_path = path_substract(target_directory, file)
            if file.suffix == ".zip":
                continue
            if not str(rel_path) in PIANO_KERN_SCORES:
                file.unlink(missing_ok=True)
    shell(
        f"cd {target_directory}\n"
        "find . -depth -type d -empty -delete\n"
    )


ASAP_MERGES = [
    ("Bach/Fugue/bwv_846/xml_score.krn", "bach/fugue/bwv_846/bwv_846.krn"),
    ("Bach/Fugue/bwv_848/xml_score.krn", "bach/fugue/bwv_848/bwv_848.krn"),
    ("Bach/Fugue/bwv_854/xml_score.krn", "bach/fugue/bwv_854/bwv_854.krn"),
    ("Bach/Fugue/bwv_856/xml_score.krn", "bach/fugue/bwv_856/bwv_856.krn"),
    ("Bach/Fugue/bwv_857/xml_score.krn", "bach/fugue/bwv_857/bwv_857.krn"),
    ("Bach/Fugue/bwv_858/xml_score.krn", "bach/fugue/bwv_858/bwv_858.krn"),
    ("Bach/Fugue/bwv_860/xml_score.krn", "bach/fugue/bwv_860/bwv_860.krn"),
    ("Bach/Fugue/bwv_862/xml_score.krn", "bach/fugue/bwv_862/bwv_862.krn"),
    ("Bach/Fugue/bwv_863/xml_score.krn", "bach/fugue/bwv_863/bwv_863.krn"),
    ("Bach/Fugue/bwv_864/xml_score.krn", "bach/fugue/bwv_864/bwv_864.krn"),
    ("Bach/Fugue/bwv_865/xml_score.krn", "bach/fugue/bwv_865/bwv_865.krn"),
    ("Bach/Fugue/bwv_866/xml_score.krn", "bach/fugue/bwv_866/bwv_866.krn"),
    ("Bach/Fugue/bwv_867/xml_score.krn", "bach/fugue/bwv_867/bwv_867.krn"),
    ("Bach/Fugue/bwv_870/xml_score.krn", "bach/fugue/bwv_870/bwv_870.krn"),
    ("Bach/Fugue/bwv_873/xml_score.krn", "bach/fugue/bwv_873/bwv_873.krn"),
    ("Bach/Fugue/bwv_874/xml_score.krn", "bach/fugue/bwv_874/bwv_874.krn"),
    ("Bach/Fugue/bwv_875/xml_score.krn", "bach/fugue/bwv_875/bwv_875.krn"),
    ("Bach/Fugue/bwv_876/xml_score.krn", "bach/fugue/bwv_876/bwv_876.krn"),
    ("Bach/Fugue/bwv_880/xml_score.krn", "bach/fugue/bwv_880/bwv_880.krn"),
    ("Bach/Fugue/bwv_883/xml_score.krn", "bach/fugue/bwv_883/bwv_883.krn"),
    ("Bach/Fugue/bwv_884/xml_score.krn", "bach/fugue/bwv_884/bwv_884.krn"),
    ("Bach/Fugue/bwv_885/xml_score.krn", "bach/fugue/bwv_885/bwv_885.krn"),
    ("Bach/Fugue/bwv_887/xml_score.krn", "bach/fugue/bwv_887/bwv_887.krn"),
    ("Bach/Fugue/bwv_888/xml_score.krn", "bach/fugue/bwv_888/bwv_888.krn"),
    ("Bach/Fugue/bwv_889/xml_score.krn", "bach/fugue/bwv_889/bwv_889.krn"),
    ("Bach/Fugue/bwv_891/xml_score.krn", "bach/fugue/bwv_891/bwv_891.krn"),
    ("Bach/Fugue/bwv_892/xml_score.krn", "bach/fugue/bwv_892/bwv_892.krn"),
    ("Bach/Fugue/bwv_893/xml_score.krn", "bach/fugue/bwv_893/bwv_893.krn"),
    ("Bach/Italian_concerto/xml_score.krn",
     "bach/italian_concerto/italian_concerto.krn"),
    ("Bach/Prelude/bwv_846/xml_score.krn", "bach/prelude/bwv_846/bwv_846.krn"),
    ("Bach/Prelude/bwv_848/xml_score.krn", "bach/prelude/bwv_848/bwv_848.krn"),
    ("Bach/Prelude/bwv_854/xml_score.krn", "bach/prelude/bwv_854/bwv_854.krn"),
    ("Bach/Prelude/bwv_856/xml_score.krn", "bach/prelude/bwv_856/bwv_856.krn"),
    ("Bach/Prelude/bwv_857/xml_score.krn", "bach/prelude/bwv_857/bwv_857.krn"),
    ("Bach/Prelude/bwv_858/xml_score.krn", "bach/prelude/bwv_858/bwv_858.krn"),
    ("Bach/Prelude/bwv_862/xml_score.krn", "bach/prelude/bwv_862/bwv_862.krn"),
    ("Bach/Prelude/bwv_863/xml_score.krn", "bach/prelude/bwv_863/bwv_863.krn"),
    ("Bach/Prelude/bwv_864/xml_score.krn", "bach/prelude/bwv_864/bwv_864.krn"),
    ("Bach/Prelude/bwv_865/xml_score.krn", "bach/prelude/bwv_865/bwv_865.krn"),
    ("Bach/Prelude/bwv_867/xml_score.krn", "bach/prelude/bwv_867/bwv_867.krn"),
    ("Bach/Prelude/bwv_868/xml_score.krn", "bach/prelude/bwv_868/bwv_868.krn"),
    ("Bach/Prelude/bwv_870/xml_score.krn", "bach/prelude/bwv_870/bwv_870.krn"),
    ("Bach/Prelude/bwv_873/xml_score.krn", "bach/prelude/bwv_873/bwv_873.krn"),
    ("Bach/Prelude/bwv_875/xml_score.krn", "bach/prelude/bwv_875/bwv_875.krn"),
    ("Bach/Prelude/bwv_876/xml_score.krn", "bach/prelude/bwv_876/bwv_876.krn"),
    ("Bach/Prelude/bwv_880/xml_score.krn", "bach/prelude/bwv_880/bwv_880.krn"),
    ("Bach/Prelude/bwv_883/xml_score.krn", "bach/prelude/bwv_883/bwv_883.krn"),
    ("Bach/Prelude/bwv_884/xml_score.krn", "bach/prelude/bwv_884/bwv_884.krn"),
    ("Bach/Prelude/bwv_885/xml_score.krn", "bach/prelude/bwv_885/bwv_885.krn"),
    ("Bach/Prelude/bwv_887/xml_score.krn", "bach/prelude/bwv_887/bwv_887.krn"),
    ("Bach/Prelude/bwv_888/xml_score.krn", "bach/prelude/bwv_888/bwv_888.krn"),
    ("Bach/Prelude/bwv_889/xml_score.krn", "bach/prelude/bwv_889/bwv_889.krn"),
    ("Bach/Prelude/bwv_891/xml_score.krn", "bach/prelude/bwv_891/bwv_891.krn"),
    ("Bach/Prelude/bwv_892/xml_score.krn", "bach/prelude/bwv_892/bwv_892.krn"),
    ("Bach/Prelude/bwv_893/xml_score.krn", "bach/prelude/bwv_893/bwv_893.krn"),
    ("Balakirev/Islamey/xml_score.krn", "balakirev/islamey/islamey.krn"),
    ("Beethoven/Piano_Sonatas/1-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/11-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/11-3/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/12-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/14-3/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/15-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/15-4/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/16-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/17-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/17-1_no_repeat/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/17-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/17-3/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/18-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/18-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/18-2_no_repeat/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/18-3/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/18-4/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/21-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/21-1_no_repeat/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/21-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/21-3/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/22-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/22-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/24-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/24-1_no_2_repeat/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/24-1_no_repeat/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/26-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/26-1_no_repeat/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/26-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/27-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/27-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/28-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/28-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/29-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/29-3/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/3-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/3-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/31-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/31-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/32-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/4-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/5-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/7-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/7-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/7-3/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/7-4/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/8-1/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/8-2/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/8-3/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/9-2_no_trio/xml_score.krn", ""),
    ("Beethoven/Piano_Sonatas/9-3/xml_score.krn", ""),
    ("Chopin/Ballades/1/xml_score.krn", "chopin/ballades/ballade-1.krn"),
    ("Chopin/Ballades/2/xml_score.krn", "chopin/ballades/ballade-2.krn"),
    ("Chopin/Ballades/3/xml_score.krn", "chopin/ballades/ballade-3.krn"),
    ("Chopin/Ballades/4/xml_score.krn", "chopin/ballades/ballade-4.krn"),
    ("Chopin/Barcarolle/xml_score.krn", "chopin/barcarolle/barcarolle.krn"),
    ("Chopin/Berceuse_op_57/xml_score.krn",
     "chopin/berceuses/berceuse57.krn"),

    ("Chopin/Etudes_op_10/1/xml_score.krn",
     "chopin/etudes/op10/etude10-1.krn"),
    ("Chopin/Etudes_op_10/10/xml_score.krn",
     "chopin/etudes/op10/etude10-10.krn"),
    ("Chopin/Etudes_op_10/12/xml_score.krn",
     "chopin/etudes/op10/etude10-12.krn"),
    ("Chopin/Etudes_op_10/3/xml_score.krn",
     "chopin/etudes/op10/etude10-3.krn"),
    ("Chopin/Etudes_op_10/4/xml_score.krn",
     "chopin/etudes/op10/etude10-4.krn"),
    ("Chopin/Etudes_op_10/5/xml_score.krn",
     "chopin/etudes/op10/etude10-5.krn"),
    ("Chopin/Etudes_op_10/7/xml_score.krn",
     "chopin/etudes/op10/etude10-7.krn"),
    ("Chopin/Etudes_op_10/8/xml_score.krn",
     "chopin/etudes/op10/etude10-8.krn"),
    ("Chopin/Etudes_op_25/1/xml_score.krn",
     "chopin/etudes/op25/etude25-1.krn"),
    ("Chopin/Etudes_op_25/10/xml_score.krn",
     "chopin/etudes/op25/etude25-10.krn"),
    ("Chopin/Etudes_op_25/11/xml_score.krn",
     "chopin/etudes/op25/etude25-11.krn"),
    ("Chopin/Etudes_op_25/12/xml_score.krn",
     "chopin/etudes/op25/etude25-12.krn"),
    ("Chopin/Etudes_op_25/2/xml_score.krn",
     "chopin/etudes/op25/etude25-2.krn"),
    ("Chopin/Etudes_op_25/4/xml_score.krn",
     "chopin/etudes/op25/etude25-4.krn"),
    ("Chopin/Etudes_op_25/5/xml_score.krn",
     "chopin/etudes/op25/etude25-5.krn"),
    ("Chopin/Etudes_op_25/8/xml_score.krn",
     "chopin/etudes/op25/etude-8.krn"),
    ("Chopin/Polonaises/53/xml_score.krn", "chopin/polonaises/polonaise_53.krn"),

    ("Chopin/Scherzos/20/xml_score.krn", "chopin/scherzos/scherzos_20.krn"),
    ("Chopin/Scherzos/31/xml_score.krn", "chopin/scherzos/scherzos_31.krn"),
    ("Chopin/Scherzos/39/xml_score.krn", "chopin/scherzos/scherzos_39.krn"),

    ("Chopin/Sonata_2/1st_no_repeat/xml_score.krn",
     "chopin/sonata/op35/sonata2-1.krn"),
    ("Chopin/Sonata_2/2nd/xml_score.krn",
     "chopin/sonata/op35/sonata2-2.krn"),
    ("Chopin/Sonata_2/2nd_no_repeat/xml_score.krn", ""),
    ("Chopin/Sonata_2/3rd/xml_score.krn", "chopin/sonata/op35/sonata2-3.krn"),
    ("Chopin/Sonata_2/4th/xml_score.krn", "chopin/sonata/op35/sonata2-4.krn"),


    ("Chopin/Sonata_3/2nd/xml_score.krn", "chopin/sonata/op58/sonata3-2.krn"),
    ("Chopin/Sonata_3/4th/xml_score.krn", "chopin/sonata/op/sonata3-4.krn"),

    ("Debussy/Images_Book_1/1_Reflets_dans_lEau/xml_score.krn",
     "debussy/reflets_dans_leau.krn"),
    ("Debussy/Pour_le_Piano/1/xml_score.krn", "debussy/pour_le_piano.krn"),

    ("Glinka/The_Lark/xml_score.krn", "glinka/the_lark/the_lark.krn"),

    ("Haydn/Keyboard_Sonatas/31-1/xml_score.krn",
     "haydn/sonatas/sonata31-1.krn"),

    ("Haydn/Keyboard_Sonatas/32-1/xml_score.krn",
     "haydn/sonatas/sonata32-1.krn"),

    ("Haydn/Keyboard_Sonatas/39-1/xml_score.krn",
     "haydn/sonatas/sonata39-1.krn"),
    ("Haydn/Keyboard_Sonatas/39-2/xml_score.krn",
     "haydn/sonatas/sonata39-2.krn"),
    ("Haydn/Keyboard_Sonatas/39-3/xml_score.krn",
     "haydn/sonatas/sonata39-3.krn"),
    ("Haydn/Keyboard_Sonatas/46-1/xml_score.krn",
     "haydn/sonatas/sonata46-1.krn"),
    ("Haydn/Keyboard_Sonatas/48-1/xml_score.krn",
     "haydn/sonatas/sonata48-1.krn"),
    ("Haydn/Keyboard_Sonatas/48-2/xml_score.krn",
     "haydn/sonatas/sonata48-2.krn"),
    ("Haydn/Keyboard_Sonatas/49-1/xml_score.krn",
     "haydn/sonatas/sonata49-1.krn"),
    ("Haydn/Keyboard_Sonatas/6-1/xml_score.krn",
     "haydn/sonatas/sonata6-1.krn"),

    ("Liszt/Annees_de_pelerinage_2/1_Gondoliera/xml_score.krn",
     "liszt/annees_de_pelerinage/gondoliera.krn"),
    ("Liszt/Ballade_2/xml_score.krn", "liszt/ballade_2/ballade_2.krn"),
    ("Liszt/Concert_Etude_S145/1/xml_score.krn",
     "liszt/etudes/s145/etude1.krn"),
    ("Liszt/Concert_Etude_S145/2/xml_score.krn",
     "liszt/etudes/s145/etude2.krn"),

    ("Liszt/Gran_Etudes_de_Paganini/2_La_campanella/xml_score.krn",
     "liszt/etudes/paganini/la_campanella.krn"),
    ("Liszt/Gran_Etudes_de_Paganini/6_Theme_and_Variations/xml_score.krn",
     "liszt/etudes/paganini/theme_and_variations.krn"),

    ("Liszt/Transcendental_Etudes/1/xml_score.krn",
     "liszt/etudes/transcendental/etude1.krn"),
    ("Liszt/Transcendental_Etudes/10/xml_score.krn",
     "liszt/etudes/transcendental/etude10.krn"),
    ("Liszt/Transcendental_Etudes/11/xml_score.krn",
     "liszt/etudes/transcendental/etude11.krn"),
    ("Liszt/Transcendental_Etudes/3/xml_score.krn",
     "liszt/etudes/transcendental/etude3.krn"),
    ("Liszt/Transcendental_Etudes/4/xml_score.krn",
     "liszt/etudes/transcendental/etude4.krn"),
    ("Liszt/Transcendental_Etudes/5/xml_score.krn",
     "liszt/etudes/transcendental/etude5.krn"),
    ("Liszt/Transcendental_Etudes/9/xml_score.krn",
     "liszt/etudes/transcendental/etude9.krn"),

    ("Liszt/Hungarian_Rhapsodies/6/xml_score.krn",
     "liszt/hungarian_rhapsodies/rhapsodies6.krn"),
    ("Liszt/Mephisto_Waltz/xml_score.krn",
     "liszt/mephisto_waltz/mephisto_waltz.krn"),
    ("Liszt/Sonata/xml_score.krn", "liszt/sonata/sonata.krn"),

    ("Mozart/Fantasie_475/xml_score.krn",
     "mozart/fantasies/fantasie475.krn"),


    ("Mozart/Piano_Sonatas/11-3/xml_score.krn", ""),
    ("Mozart/Piano_Sonatas/12-1/xml_score.krn", ""),
    ("Mozart/Piano_Sonatas/12-2/xml_score.krn", ""),
    ("Mozart/Piano_Sonatas/12-3/xml_score.krn", ""),
    ("Mozart/Piano_Sonatas/8-1/xml_score.krn", ""),

    ("Prokofiev/Toccata/xml_score.krn", "prokofiev/toccata/toccata.krn"),

    ("Rachmaninoff/Preludes_op_23/4/xml_score.krn",
     "rachmaninoff/preludes/prelude23-4.krn"),
    ("Rachmaninoff/Preludes_op_23/6/xml_score.krn",
     "rachmaninoff/preludes/prelude23-6.krn"),
    ("Rachmaninoff/Preludes_op_32/5/xml_score.krn",
     "rachmaninoff/preludes/prelude32-5.krn"),

    ("Ravel/Gaspard_de_la_Nuit/1_Ondine/xml_score.krn",
     "ravel/gaspard_de_la_nuit/ondine.krn"),
    ("Ravel/Miroirs/3_Une_Barque/xml_score.krn",
     "ravel/miroirs/une_barque.krn"),
    ("Ravel/Miroirs/4_Alborada_del_gracioso/xml_score.krn",
     "ravel/miroirs/alborada_del_gracioso.krn"),
    ("Ravel/Pavane/xml_score.krn", "ravel/pavane/pavane.krn"),

    ("Schubert/Impromptu_op.90_D.899/1/xml_score.krn",
     "schubert/impromptu/op90/impromptu-1.krn"),
    ("Schubert/Impromptu_op.90_D.899/2/xml_score.krn",
     "schubert/impromptu/op.90/impromptu-2.krn"),
    ("Schubert/Impromptu_op.90_D.899/3/xml_score.krn",
     "schubert/impromptu/op.90/impromptu-3.krn"),
    ("Schubert/Impromptu_op.90_D.899/4/xml_score.krn",
     "schubert/impromptu/op.90/impromptu-4.krn"),

    ("Schubert/Impromptu_op142/1/xml_score.krn",
     "schubert/impromptu/op142/impromptu-1.krn"),

    ("Schubert/Moment_Musical_no_1/xml_score.krn",
     "schubert/moment_musical/moment_musical-1.krn"),
    ("Schubert/Moment_musical_no_3/xml_score.krn",
     "schubert/moment_musical/moment_musical-3.krn"),

    ("Schubert/Wanderer_fantasie/xml_score.krn",
     "schubert/wanderer_fantasie/wanderer_fantasie.krn"),

    ("Schumann/Kreisleriana/1/xml_score.krn",
     "schumann/kreisleriana/kreisleriana-1.krn"),
    ("Schumann/Kreisleriana/2/xml_score.krn",
     "schumann/kreisleriana/kreisleriana-2.krn"),
    ("Schumann/Kreisleriana/5/xml_score.krn",
     "schumann/kreisleriana/kreisleriana-5.krn"),
    ("Schumann/Kreisleriana/7/xml_score.krn",
     "schumann/kreisleriana/kreisleriana-7.krn"),

    ("Schumann/Toccata/xml_score.krn", "schumann/toccata/toccata.krn"),

    ("Schumann/Toccata_repeat/xml_score.krn",
     "schumann/toccata_repeat/toccata_repeat.krn"),

    ("Scriabin/Sonatas/5/xml_score.krn", "scriabin/sonatas/sonata5.krn"),
]

# asap duplicates, in (kernscores, matching asap) format:
# (chopin/ballade/ballade52.krn chopin/ballades/4/ballade-4.krn)


@dataclass(frozen=True)
class Entry:
    kern_file: str

    # The source dataset for this entry, "kern-score/*.zip" or "asap"
    source: str
    imslp_url: str
    pdf_urls: List[str]

    @staticmethod
    def from_dict(data):
        return Entry(**data)


@dataclass(frozen=True)
class Catalog:
    version: int = 1

    entries: Dict[str, Entry] = field(default_factory=dict)


class KernSheet:
    CATALOG_NAME = "catalog.json"

    datadir: Path
    version: int = 1

    def __init__(self, datadir: Path):
        super().__init__()
        self.datadir = Path(datadir)
        self.load_catalog()

    def load_catalog(self):
        path = self.datadir / self.CATALOG_NAME
        self.version = 1
        self.entries = {}
        if path.exists():
            with open(path, "r") as fp:
                obj = json.load(fp)
            self.version = obj["version"]
            self.entries = {kern_file: Entry.from_dict(
                entry_dict) for kern_file, entry_dict in obj["entries"].items()}

    def save_catalog(self):
        path = self.datadir / self.CATALOG_NAME
        with open(path, "w+") as fp:
            json.dump({
                "version": 1,
                "entries": {
                    k: asdict(replace(e, kern_file=str(k))) for k, e in self.entries.items()
                }
            }, fp, indent=4)

    KERN_SCORE_URL = "https://kern.humdrum.org/cgi-bin/ksdata?location=users/craig/classical/"

    def kernscore_url(self, kern_file: Path) -> str:
        return (
            self.KERN_SCORE_URL + str(kern_file.parent) +
            f"&file={quote(kern_file.name)}&format=pdf"
        )

    def missing(self):
        """Checks for orphaned .krn files with no entries in the catalog.
        """
        total_count = 0
        miss_count = 0
        for root, _, filenames in os.walk(self.datadir):
            for filename in filenames:
                file = Path(root) / filename
                if file.suffix == ".krn":
                    total_count += 1
                    kern_file = path_substract(self.datadir, file)
                    if not str(kern_file) in self.entries:
                        miss_count += 1
        print(f"{total_count} kern files, {miss_count} missing.")

    KERN_KEYWORDS_RE = re.compile(r'^!!!(COM|OPR|OTL|OPS):\s*(.*)$')

    def google_keywords(self, kern_file: str) -> str:
        keywords = ""
        # Checks inside the file for COM and OTL:
        with open(self.datadir / kern_file, "r") as fp:
            for line in fp:
                line = line.strip()
                if not line.startswith("!!!"):
                    break
                elif (m := self.KERN_KEYWORDS_RE.match(line)):
                    keywords += f" {m.group(2).lower()}"
        # Adds in the path components:
        path_str = str(Path(kern_file).with_suffix(""))
        keywords += " " + re.sub(r'[^\w]+', " ", path_str.lower()).strip()
        return " ".join(keywords.split()) + " site:imslp.org"

    def fix_imslp(self):
        imslp = IMSLP()
        for key, entry in self.entries.items():
            if entry.imslp_url:
                continue
            logging.info(f"+ fix_imslp: {key}")
            imslp_url = imslp.find_imslp(self.google_keywords(entry.kern_file))
            if imslp_url is not None:
                self.entries[key] = replace(entry, imslp_url=imslp_url)
                self.save_catalog()
            time.sleep(30)

    def staff(self, kern_path: str) -> List[Tuple[MatLike, Staffer.Page]]:
        path = self.datadir / kern_path
        pkl_path = path.with_suffix(".pkl")
        if path.with_suffix(".pkl").exists():
            with open(pkl_path, "rb") as fp:
                data = cast(List[Tuple[MatLike, Staffer.Page]],
                            pickle.load(fp))
        else:
            staffer = Staffer(path)
            data = staffer.staff()
            with open(pkl_path, "wb+") as fp:
                pickle.dump(data, fp)
        return data


@click.command()
@click.argument("asap", required=False,
                type=click.Path(file_okay=False, dir_okay=True, exists=True),
                default="/home/anselm/Downloads/asap-dataset")
@click.argument("kern_sheet", required=False,
                type=click.Path(file_okay=False, dir_okay=True, exists=True),
                default="/home/anselm/datasets/kern-sheet")
def merge_asap(asap: Path, kern_sheet: Path):
    asap = Path(asap)
    # Converts all .musicxml files into .krn files using verovio.
    processed = 0
    for root, _, filenames in os.walk(asap):
        for filename in filenames:
            file = Path(root) / filename
            if file.suffix == ".musicxml":
                # processed += verovio(file)
                pass
    print(f"Translated {processed} .musicxml to .krn files")
    # Move the kern files to the right spot, that's the easiest way to go :(
    for src, dst in ASAP_MERGES:
        if not dst:
            continue
        src, dst = asap / src, kern_sheet / Path(dst)
        if src.exists():
            print(f"{src} => {dst}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(src, dst)
        else:
            print(f"ASAP_MERGES: {src} not found.")


@click.command()
@click.argument("datadir",
                type=click.Path(dir_okay=True, exists=True),
                default="/home/anselm/datasets/kern-sheet/")
def do(datadir: Path):
    kern_sheet = KernSheet(Path(datadir))
    kern_sheet.fix_imslp()


@click.command()
@click.argument("datadir",
                type=click.Path(file_okay=True, dir_okay=True),
                default="/home/anselm/datasets/kern-sheet/")
@click.argument("kern_path", type=str, required=True)
def staff(datadir: Path, kern_path: str):
    kern_sheet = KernSheet(datadir)
    kern_sheet.staff(kern_path)


@click.command()
@click.argument("datadir",
                type=click.Path(dir_okay=True, exists=True),
                default="/home/anselm/datasets/kern-sheet/")
def missing(datadir: Path):
    kern_sheet = KernSheet(Path(datadir))
    kern_sheet.missing()


@click.group
def cli():
    pass


cli.add_command(make_kern_sheet)
cli.add_command(merge_asap)
cli.add_command(do)
cli.add_command(staff)
cli.add_command(missing)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    cli()
