Experiments of Online Item Recommenders
===

## Requirements

- Python
- numpy
- scipy
- scikit-learn

## Usage

	$ python experiment.py -f path/to/config/file.ini
	
Examples of config files are available at: [config/](config/)

The results will be written text files under *results/*.

## References

### Incremental Matrix Factorization

J. Vinagre et al. **[Fast Incremental Matrix Factorization for Recommendation with Positive-only ](http://link.springer.com/chapter/10.1007/978-3-319-08786-3_41)**. In *Proc. ofUMAP 2014*, pp. 459–470, July 2014.

- My early implementation: [takuti/incremental-sgd](https://github.com/takuti/incremental-sgd)

### Incremental Factorization Machines

Takuya Kitazawa. **[Incremental Factorization Machines for Item Recommendation in Data Streams](https://kaigi.org/jsai/webprogram/2016/paper-170.html)**. *[The 30th Annual Conference of the Japanese Society for Artificial Intelligence](http://www.ai-gakkai.or.jp/jsai2016/)*, 1C2-5, June 2016.

- My implementation of the static factorization machines: [takuti/factorization-machines](https://github.com/takuti/factorization-machines)

### Sketching Positive Feedback

Takuya Kitazawa. **Incremental Item Recommendation Using a SVD-based Streaming Anomaly Detection Framework** (in Japanese). *[Numerical Analysis Symposium 2016](http://www.nas.sr3.t.u-tokyo.ac.jp/)*, June 2016.

- Feasibility of frequent directions is tested on: [takuti/incremental-matrix-approximation](https://github.com/takuti/incremental-matrix-approximation)
- My implementation of the original streaming anomaly detection framework: [takuti/stream-anomaly-detect](https://github.com/takuti/stream-anomaly-detect)