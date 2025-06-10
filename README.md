# Connectivity_EEG_benchmarking
This repository contains code associated with my thesis 'Comparing functional and effective connectivity measures for EEG'. This work is still in progress. Estimated date of completion: September 2025.

## Abstract  
### Background:  
Brain connectivity measures have been used to study communication between brain regions using electroencephalography (EEG). Functional and effective connectivity estimate the synchronization and the flow of information between regions, respectively. However, findings from studies using different measures to investigate similar connections do not converge.
To guide the selection of functional and effective connectivity measures in future studies, we systematically compared a set of measures in the context of resting state EEG. We examined four functional connectivity metrics (coherence (Coh), the imaginary part of coherence (imCoh), the corrected imaginary part of phase lagged value (ciPLV), the debiased weighted phase-locking index (dwPLI)) and three effective connectivity measures (generalized partial directed coherence (gPDC), direct directed transfer function (dDTF), and pairwise spectral granger prediction (pSGP)).  

### Methods:  
We compared the measures using non-dynamic and dynamic models of simulated EEG to assess their performance in the presence of three confounders: i) common input, ii) indirect connections, and iii) volume conduction. We also used experimental EEG to calculate i) the effect size of the difference between eyes-closed (EC) and eyes-open (EO) resting-state conditions and ii) the ratio of the inter-to-intra-subject variance in connectivity estimates.  


### Results:  
Among the functional connectivity measures, we observed that dwPLI is the least sensitive to volume conduction. dwPLI, along with ciPLV, better distinguished EC and EO in experimental EEG (p<0.01). However, Coh displayed the highest inter-to-intra-subject variance ratio, meaning that Coh would more likely produce consistent measurements from individual versus from different individuals. 
We observed that dDTF is the least sensitive to volume conduction. dDTF also best distinguished between EC and EO conditions, suggesting that it is better suited for analyses between two experimental conditions (p's < 0.05).  


### Discussion:  
Some of our results contradicted our hypotheses. Among the functional connectivity measures, Coh was less sensitive to common input and had a higher between-to-within subject variance ratio, compared to dwPLI. Among the effective connectivity measures, pSGP had the highest between-to-within subject variance ratio. We discuss potential explanations for these results and ways to further validate them.  


### Conclusion:  
Our results allow us to make a few recommendations regarding connectivity measurement. Researchers might use dwPLI if they are interested in functional brain connectivity that is not instantaneous. Because Coh performed better than dwPLI against common input, we suggest that a measurement with Coh be performed alongside, followed by correction for volume conduction using techniques such as partialization. 
Researchers might use dDTF if they are interested in effective brain connectivity that is not instantaneous. Because gPDC performed similarly to dDTF against indirect connections and common input, we recommend that gPDC be used alongside dDTF.  
