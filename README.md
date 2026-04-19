# NL-NMC

Project NL-NMC: nonlinear Monte Carlo framework for modeling light–matter interactions in turbid media

Light propagation in turbid media such as biological tissues involves complex multiple scattering processes that are further complicated by nonlinear optical effects. While the Monte Carlo method is widely regarded as the gold standard for modeling photon transport, extending it to nonlinear regimes introduces additional computational and algorithmic challenges. NL-NMC is an open-source Monte Carlo framework designed to address these challenges by enabling efficient simulation of both linear and nonlinear light-matter interactions. The method builds upon energy-efficient GPU architectures, with a particular focus on Apple Silicon, and introduces advanced memory handling and synchronization strategies required for modeling photon–photon coupling effects such as stimulated Raman scattering. In addition to conventional outputs like fluence and reflectance, NL-NMC provides spatially and temporally resolved radiance maps, separated by photon type, as well as detailed photon trajectory data. 

The framework is implemented as a flexible and extensible software package, allowing adaptation to a wide range of biophotonics and spectroscopic applications.
