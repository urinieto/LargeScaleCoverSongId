"""Downloads the preview MP3 from 7digital given an Echo Nest Track ID.

Requires the pyechonest module.

Created by Uri Nieto, 2013"""

import argparse
import urllib2
from pyechonest import config
from pyechonest import song

# API Parameters
config.ECHO_NEST_API_KEY="RRIGOW0BG5NJTM5M4" # Uri API Key
config.ECHO_NEST_CONSUMER_KEY = "3cbd6b002e3594724be404d22dcd7c72"
config.ECHO_NEST_SHARED_SECRET = "4TPdu5YASG2hDe7zUf2ZdA"

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