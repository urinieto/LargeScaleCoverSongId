#!/usr/bin/env python
"""Downloads the preview MP3 from 7digital given an Echo Nest Track ID.

Requires the pyechonest module.

----
Authors: 
Uri Nieto (oriol@nyu.edu)

----
License:
This code is distributed under the GNU LESSER PUBLIC LICENSE 
(LGPL, see www.gnu.org).

Copyright (c) 2012-2013 MARL@NYU.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

  a. Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.
  b. Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.
  c. Neither the name of MARL, NYU nor the names of its contributors
     may be used to endorse or promote products derived from this software
     without specific prior written permission.


THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.
"""

import argparse
import urllib2
from pyechonest import config
from pyechonest import song

# API Parameters
config.ECHO_NEST_API_KEY="" # TODO: Add your keys!
config.ECHO_NEST_CONSUMER_KEY = ""
config.ECHO_NEST_SHARED_SECRET = ""

def download(url, localName):
    """Downloads the file from the url and saves it as localName."""
    req = urllib2.Request(url)
    r = urllib2.urlopen(req)
    f = open(localName, 'wb')
    f.write(r.read())
    f.close()

def get_url_from_track_id(track_id):
    s = song.profile(track_ids=track_id)
    assert len(s) > 0, "ERROR: Track not found in The Echo Nest!"
    track = s[0].get_tracks("7digital-US")
    assert len(track) >0, "ERROR: Track not found in 7digital!"
    url = track[0]["preview_url"]
    return url

def process(track_id):
    """Finds the URL and downloads it into the local dir."""
    url = get_url_from_track_id(track_id)
    print "Downloading from", url
    download(url, track_id + ".mp3")

def main():
    # Args parser
    parser = argparse.ArgumentParser(description=
                "Downloads an mp3 given an Echo Nest Track ID")
    parser.add_argument("track_id", action="store", help="Echo Nest Track ID")
    args = parser.parse_args()
    process(args.track_id)
    print 'DONE!'

if __name__ == '__main__':
    main()