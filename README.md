## Quantum Deep Policy Gradient Reinforcement Learning
This is the source code of our paper [Quantum DRL-Based UAV Placement with Optical Sensing-Enabled Hybrid FSO/RF Backhaul Design](https://ieeexplore.ieee.org/abstract/document/11742613) published IEEE VTS APWCS 2026 proceedings, proposing the Quantum Deep Policy Gradient for UAV Placement in dynamic networks.
> This study introduces an architecture integrating optical sensing-enabled hybrid free-space optics (FSO)/radiofrequency (RF) backhaul networks with unmanned aerial vehicle (UAV)-mounted base stations to provide high-speed connectivity for ground mobile users (MUs). To maximize the number of served high-data-rate MUs, a backhaul link-switching scheme is adopted to switch between FSO and RF links under varying environmental conditions, while a quantum deep reinforcement learning (QDRL) model is proposed for optimal UAV placement. Simulation results demonstrate the effectiveness of the linkswitching scheme in maintaining stable backhaul connectivity under dynamic environmental conditions. Notably, the proposed QDRL outperforms classical DRL benchmarks in terms of learning stability and parameter efficiency, highlighting its potential to provide stable control policies and scalable solutions for dynamic and large-scale networks.
### Train our proposed Quantum Deep Policy Gradient model🤖
~~~
py QuantumDRL_MultiUAV.py
~~~
### Test our proposed Quantum Deep Policy Gradient model🌗
~~~
py QDRL_testing.py
~~~

### Paper citation
~~~
@inproceedings{le2026quantum,
  title={Quantum DRL-Based UAV Placement with Optical Sensing-Enabled Hybrid FSO/RF Backhaul Design},
  author={Le, Hung V and Le, Hoang D and Dao, Duy-Tuan and Pham, Anh T},
  booktitle={2026 IEEE VTS Asia Pacific Wireless Communications Symposium (APWCS)},
  pages={32--36},
  year={2026},
  organization={IEEE}
}
~~~