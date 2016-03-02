Incremental Factorization Machines
===

Takuya Kitazawa. **Incremental Factorization Machines for Item Recommendation in Data Streams**. In *Proc. of the 30th National Convention of the Japanese Society for Artificial Intelligence*, 1C2-5, June 2016 (to appear). 

## Usage

	$ python experiment.py --model=MODEL
	
***MODEL*** can be

- **baseline:** incremental matrix factorization w/o updating
- **iMF:** incremental matrix factorization
- **biased-iMF:** biased incremental matrix factorization
- **iFMs:** incremental factorization machines *(proposed)*
- **iFMs-time-aware:** time-awared incremental factorization machines *(proposed)*
- **all_MF:** run **baseline**, **iMF** and **biased-iMF** at once

The results will be written text files under *results/*.