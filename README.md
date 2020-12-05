This is a Rust implementation of the technique of "Reconstructability
Analysis". I used these sources for information about how it should
work:

- Zwick, [An Overview of Reconstructability Analysis][overview], 2004
- Zwick, [Wholes and Parts in General Systems Methodology][wholes], 2001
- the existing C++ implementation, [OCCAM][]
- the [OCCAM manual][]

[overview]: https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1022&context=sysc_fac
[wholes]: https://pdxscholar.library.pdx.edu/cgi/viewcontent.cgi?article=1026&context=sysc_fac
[OCCAM]: https://github.com/occam-ra/occam
[OCCAM manual]: https://occam.readthedocs.io/en/latest/

And, as always, credit goes to a pile of Wikipedia articles and other
sources to familiarize myself with the underlying statistics principles.

# Status

I believe this program computes correct answers on small input. The
search space is doubly exponential in the number of variables, and this
program currently just exhaustively evaluates the whole search space, so
I expect it to fall over pretty quickly as problem sizes get bigger.

There are also various places that can overflow even 64-bit floats, such
as computing the number of degrees of freedom for any problem with at
least 1,024 variables. But, mixed blessings, it'll fail for the previous
reasons before you can hit this problem in practice.

On the other hand, I think my implementation of many of the fundamental
operations should be faster both asymptotically and in terms of constant
factors than the existing C++ implementation. It's also much less source
code and, I think, easier to understand.

I wrote this because I wanted to better understand the technique and
also for fun. I don't know yet whether I'll do more with it.

# Future work

There are so many opportunities for parallelism (perhaps with [rayon][])
and caching which should at least provide constant-factor improvements.
And while I've tried to be careful about memory use and allocation
rates, I haven't done any profiling so you can probably find savings all
over.

[rayon]: https://crates.io/crates/rayon

Normalizing a model so that no relation in it is a subset of any other
is non-trivial, and when I asked friends about how to do better, one
pointed me at the following papers. I've only skimmed them but they seem
promising:

- Bayardo & Panda, "[Fast algorithms for finding extremal
  sets][fast-extremal]", SIAM Intl Conf on Data Mining 2011
- Marinov, Nash & Gregg, "[Practical algorithms for finding extremal
  sets][practical-extremal]", ACM Journal of Experimental Algorithmics,
  2016

[fast-extremal]: http://bayardo.org/ps/sdm2011.pdf
[practical-extremal]: https://arxiv.org/pdf/1508.01753.pdf

As with many projects, more tests would be helpful. I think this is an
especially good candidate for [property testing][] and perhaps also
[mutation testing][].

[property testing]: https://crates.io/crates/proptest
[mutation testing]: https://github.com/llogiq/mutagen
