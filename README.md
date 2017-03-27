# GraBTax
Graph-based, query-dependent taxonomy generation library

GraBTax (Graph-based Automatic Taxonomy Generation) is a query-dependent taxonomy 
generation approach proposed in (Treeratpituk et al., 2013), for constructing query-dependent taxonomies from a weighted graph representing 
topic associations present within a text corpus. A graph partitioning algorithm is used to recursively partition the topic graph 
into taxonomies, and partition labels are selected based on network centrality within partitioned subgraphs (Treeratpituk et al., 2013).
  
This library is implemented using the multilevel graph partitioning scheme discussed in 
(Karpis and Kumar, 1999), and implemented in [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) (redistributed in this repo under the original Apache 2.0 license). Additionally, [METIS for Python](https://metis.readthedocs.io/en/latest/) is used 
to wrap the METIS library, which is licensed by [Ken Watford](https://bitbucket.org/kw/) under MIT and also redistributed here.
  
### Overview
  
  ...
  
### Licensing
METIS is redistributed courtesy of [Regents of the University of Minnesota and Karypis Lab](http://glaros.dtc.umn.edu/gkhome/) under the Apache 2.0 license, which can be viewed [here](metis-5.1.0/LICENSE.txt). METIS for Python is redistributed courtesy 
of [Ken Watford](https://bitbucket.org/kw/) under the MIT License, which can be viewed [here](metis-python/LICENSE.txt). All other sources, unless otherwise stated, are licensed under [Apache 2.0](LICENSE.txt)

\[Treeratpituk et al.2013\] P Treeratpituk, M Khabsa, and CL Giles.  2013  Graph-based Approach to Automatic Taxonomy Generation (GraBTax)  _arXiv:1307.1718v1 \[cs.IR]_\](https://arxiv.org/abs/1307.1718v1)  
\[Karypis and Kumar1999\] G Karypis and V Kumar.  1999  A fast and high quality multilevel scheme for partitioning irregular graphs. _SIAM Journal on Scientific Computing_, Vol. 20, No. 1, pp. 359—392, 1999.  


