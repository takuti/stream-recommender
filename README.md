Incremental Factorization Machines
===

Takuya Kitazawa. **Incremental Factorization Machines for Item Recommendation in Data Streams**. In *Proc. of the 30th National Convention of the Japanese Society for Artificial Intelligence*, 1C2-5, May 2016 (to appear).

## Usage

	$ python experiment.py
	
- **--model**: specify a factorization model
	- **baseline** &mdash; incremental matrix factorization w/o updating
	- **iMF** &mdash; incremental matrix factorization
	- **biased-iMF** &mdash; biased incremental matrix factorization
	- **iFMs** &mdash; incremental factorization machines *(proposed)*
	- **all_MF** &mdash; run **baseline**, **iMF** and **biased-iMF** at once
	- **all_FMs** &mdash; run **iFMs** both of *w/o contextual variables* and *w/ all contextual variables*
- **--context**, **-c**: choose contexts used by iFMs
	- **dt** &mdash; elapsed days from the initial positive sample
	- **genre** &mdash; movie genre
	- **demographics** &mdash; users' demographics
- **--limit**: restrict number of test samples

Sample:

	$ python experiment.py --model=iFMs -c genre -c dt -c demographics

The results will be written text files under *results/*.