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
- **--context**: choose whether contexts are used as a feature (bool)
- **--method**: evaluation method
	- **recall** &mdash; evaluate the later 50% (T=500)
	- **monitor** &mdash; evaluate the later 70% (T=5000)
- **--dataset**
	- **ML1M** &mdash; positive samples in MovieLens 1M
	- **ML100k** &mdash; positive samples in MovieLens 100k
	- **LastFM** &mdash; subset of [the LastFM dataset](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/lastfm-1K.html)
- **--n_epoch**: number of epochs in the batch pre-training

Sample:

	$ python experiment.py --method=monitor --model=iFMs --dataset=ML1M --n_epoch=1 --context

The results will be written text files under *results/*.