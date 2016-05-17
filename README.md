Item Recommendation in Data Streams
===

Takuya Kitazawa. **Incremental Factorization Machines for Item Recommendation in Data Streams**. In *Proc. of the 30th National Convention of the Japanese Society for Artificial Intelligence*, 1C2-5, May 2016 (to appear).

## Usage

	$ python experiment.py
	
- **--model**: specify a factorization model
	- **random** &mdash; randombaseline
	- **static-MF** &mdash; incremental matrix factorization w/o updating
	- **iMF** &mdash; incremental matrix factorization
	- **iFMs** &mdash; incremental factorization machines *(proposed)*
- **--dataset**
	- **ML1M** &mdash; positive samples in MovieLens 1M
	- **ML100k** &mdash; positive samples in MovieLens 100k
	- **LastFM** &mdash; subset of [the LastFM dataset](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html)
- **--window_size**: for incremental recall (default=5000)
- **--n_epoch**: number of epochs in the batch pre-training

Samples:

	$ python experiment.py --model=random --dataset=ML1M
	$ python experiment.py --model=static-MF --dataset=ML1M --n_epoch=9
	$ python experiment.py --model=iMF --dataset=ML1M --n_epoch=9
	$ python experiment.py --model=iFMs --dataset=ML100k --n_epoch=10

The results will be written text files under *results/*.
