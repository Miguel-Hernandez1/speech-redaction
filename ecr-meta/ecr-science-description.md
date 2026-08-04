# Speech Redaction at the Edge

## The Science

Acoustic monitoring is a powerful tool for studying the natural world. To explain, a microphone left running in a forest or a park can record birdsong around the
clock, and models like BirdNET can turn those recordings into a running count
of which species are present and when. And it is this kind of continuous, passive
listening lets ecologists study animal populations at a scale that would be
impossible to reach by hand.

However, there is an issue, a microphone that is always listening does not only hear birds; it also
hears people. So when a monitoring node is placed somewhere the public can walk
past it, such as a trail in a national park, that microphone will inevitably
capture human conversation. This is a serious privacy problem, and it is the
reason land managers are often reluctant to allow always-on audio recording in
public spaces at all.

This plugin exists to solve that problem directly; its job is to remove human
speech from the audio at the edge, on the node itself, before the recording is
ever saved or sent anywhere. And the result is audio that still contains the
birdsong and the natural soundscape that scientists want, but with human
speech erased. This means that the science can continue, and the people walking past are not
recorded.

## The Motivating Case

The Sage project is deploying a monitoring node at Haleakala National Park in
Hawaii. The park runs a BirdNET pipeline to identify native and endangered
birds from their calls, which means the node needs its microphone on. The
National Park Service does not want park visitors recorded, and their simplest
option was to ask that the microphone be turned off entirely.

Turning the microphone off would also end the bird research. This plugin is
the alternative that lets both goals coexist: keep listening for birds, but
guarantee that no human speech is kept.

## How It Works

The plugin treats speech removal as something that must happen before the
audio is stored, not after. This ordering is the heart of the design. If
speech were filtered out only after a recording had already been written to
disk, the private audio would have existed, however briefly, in a saved file.
By redacting while the audio is still held in memory, the plugin ensures that
a recording containing raw human speech is never written at all.

Detection is handled by YAMNet, a general-purpose audio classifier that scores
short frames of audio for how much they sound like speech. Those scores are
fed into a decision component that decides which stretches of time to erase.
This component uses hysteresis, meaning it requires a clear signal to begin
erasing and a clear signal to stop, so that it does not switch on and off in
the middle of a spoken sentence. Each erased stretch is padded slightly on
both sides so that the very beginning and end of an utterance are never left
behind.

So where speech is found, the audio samples are set to silence. And the surrounding
sound, wind, rain, birdsong, and general ambience, is left untouched. This
is important because the goal is not to blank out the recording but to only remove
the human speech while preserving the natural soundscape that the science
depends on.

## Failing Safe

Because the privacy guarantee has to hold even when something goes wrong, the
plugin is built to fail safe. Its default behavior is to redact. Successful
speech detection is what permits a recording to be kept, rather than detection
being the thing that triggers redaction. If the speech detector fails, returns
nothing, or hits any unexpected error, the plugin erases the entire buffer
rather than risk letting speech slip through. There is no path in which a
detection failure results in raw speech being kept.

## Redaction as Its Own Record

Rather than filling erased spans with artificial noise, which would corrupt the
acoustic record that downstream analysis relies on, the design writes silence
and can publish a separate record of each redaction event, including when it
happened and how long it lasted. This makes every redaction auditable. It lets
a reviewer tell the difference between a quiet moment and a broken sensor, and
it gives land managers a verifiable log of when the system removed speech,
without ever revealing what was said.

## Status

The speech detection and redaction pipeline has been built and tested, and it
has been verified running on the target edge hardware, an NVIDIA Jetson AGX
Thor. On real speech recordings, the detector reliably separates speech from
silence, and the redaction step removes the speech while leaving the
surrounding audio intact. This plugin packages that work as a standalone Sage
application that consumes audio from the node's local cache and produces a
redacted version for downstream applications such as BirdNET to use.

## Acknowledgment

This work was supported in part by the National Science Foundation under
Awards No. 2331263 and 2436842.
