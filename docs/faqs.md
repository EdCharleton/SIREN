# FAQs (Frequently Asked Questions)



<h3>Why is the SIREN .zip file so large?</h3>
To create a standalone version of SIREN (which was written in Julia), we build a system image which contains: the compiled SIREN code, the Julia runtime (including its LLVM compiler backend) and all package dependencies which are bundled into one self-contained folder. This makes the download large (~1.4 GB), but it was a deliberate choice:

1. Zero setup: you don’t need to install Julia or manage packages or environments.
1. Faster start-up: Julia normally compiles code just-in-time the first time it is run. Shipping a pre-compiled image removes most of that delay (package loading and compilation), so SIREN starts much faster. *NOTE: The first classification will be slow as it needs to start up. But will be much faster after that.*

The bundle is built for a specific operating system and CPU architecture (currently macOS on Apple Silicon). *I am working on reducing the file size if I can.*

<h3>Why is the returned spectrum is a nonsenese fit of a featureless TDE, SLSN or Type II supernovae?</h3>

Currently SIREN doesn't have a way of saying if any of its templates are a good fit to the spectrum or not, especially in a spectrum with a lot of host contamination.  As such, it will often fit a featureless Type II, TDE or SLSN as the best fit to match the continuum but not catch any features. *This is the first item on my list of things to solve.*