Item Recommendation in a Streaming Environment
===

This repository discloses implementation used in the following research papers:

- T. Kitazawa. **[Incremental Factorization Machines for Persistently Cold-Starting Online Item Recommendation](https://arxiv.org/abs/1607.02858)**. arXiv:1607.02858 [cs.LG], July 2016.
- T. Kitazawa. **[Sketching Dynamic User-Item Interactions for Online Item Recommendation](http://dl.acm.org/citation.cfm?id=3022152)**. In *Proc. of CHIIR 2017*, March 2017.

Recommendation algorithms are implemented in [FluRS](https://github.com/takuti/flurs), a Python library for online item recommendation tasks.

## Usage

	$ python experiment.py -f path/to/config/file.ini
	
Examples of config files are available at: [config/](config/)

The results will be written text files under *results/*.