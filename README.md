# AI-Driven Indoor Flow Analytics for Smart Space Sustainability

[![Competition Track](https://img.shields.io/badge/Track-Workplace%20AI-blue)](https://github.com/)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/Model-YOLOv8s-orange)](https://github.com/ultralytics/ultralytics)

---

## 1. Background & Problem Definition
* **The Challenge**: Traditional indoor space management (e.g., access control, infrared sensors) only provides aggregate entry/exit counts without capturing dynamic movement trajectories, dwell times, or spatial utilization rates.
* **The Consequence**: This lack of behavioral and spatial data often leads to inefficient energy consumption in HVAC and lighting systems, creating blind spots in smart building management.
* **Target Scenarios**: Libraries, self-study rooms, exhibition halls, and corporate workplace environments.

---

## 2. Solution, Tools & Methodology
* **Core Spirit**: *"Transforming Pixels into Actionable Sustainability Metrics."*
* **AI Pipeline**: 
  - **Object Detection**: **YOLOv8s** is deployed to balance high detection accuracy with edge computing performance.
  - **Trajectory Tracking**: Integrated **ByteTrack** (`persist=True`) to solve occlusion and ID-switching issues, ensuring continuous movement paths.
* **7 Core Sustainability Metrics**:
  1. **Area Congestion Rate** (Aligned with ASHRAE occupant density standards)
  2. **Spatial Efficiency** (Aligned with Taiwan's Smart Building Assessment Manual)
  3. **Inefficient Space Ratio** (Quantifies long-term idle zones)
  4. **Crowding Index** (Based on E.T. Hall’s social distance theory)
  5. **Spatial Turnover Rate** (Calculates seat/zone turnover frequency)
  6. **Average Dwell Time** (Statistical analysis of occupant duration)
  7. **Circulation Exploration Degree** (Analyzes movement path richness and wayfinding behavior)

---

## 3. Hardware & Software Architecture
* **1. Data & Device Layer**: Integration of authorized indoor surveillance video streams and edge computing hardware (compatible with ASUS / Hailo-10H edge accelerators).
* **2. Logic & Calculation Layer**: Automated data processing pipeline combining YOLOv8s detection, ByteTrack tracking, and metric calculation algorithms.
* **3. Frontend Interactive Layer**: Visualized dashboard and decision-support interface for facility managers.

---

## 4. Expected Benefits & Outputs
* **Energy Optimization**: Enables dynamic, zone-based HVAC and lighting scheduling to reduce carbon emissions.
* **Operational Efficiency**: Transforms passive surveillance footage into active management intelligence.
* **Empirical Validation**: Successfully tested and validated across real-world testing environments (e.g., university libraries and commercial shopping malls).

---

## 5. Repository Structure & Quick Start
```text
ai-space-sustainability/
├── docker/               # Containerization configuration
├── libraryTestResult/    # Empirical test data and heatmaps (Library)
├── mallTestResult/       # Empirical test data and heatmaps (Shopping Mall)
├── testCode/             # Core application and analysis scripts
├── ultralytics/          # YOLO core library and modules
├── yoloModels/           # Trained model weights (.pt)
├── .gitignore            # Git exclusion rules
├── LICENSE               # Open-source license agreement
└── pyproject.toml        # Python project dependencies

```

### Quick Start

1. Clone the repository:
```bash
git clone [https://github.com/youllli/ai-space-sustainability]
cd ai-space-sustainability

```


2. Install dependencies:
```bash
pip install -r pyproject.toml

```


3. Run the analysis script:
```bash
python testCode/main.py

```



---

## 6. References

1. J. Redmon, S. Divvala, R. Girshick, and A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," Cv-foundation.org, pp. 779-788, 2016.
2. 何强, “Indoor People Counting Model Based on Improved YOLO V3 Algorithm," Hans Journal of Data Mining, vol. 13, no. 01, pp. 10-22, 2023.
3. Y. Zhao, X. Zhao, S. Chen, Z. Zhang, and X. Huang, "An Indoor Crowd Movement Trajectory Benchmark Dataset," IEEE Transactions on Reliability, vol. 70, no. 4, pp. 1368-1380, Dec. 2021.
4. J. Yuan, L. Li, E. Wang, and M. J. Skibniewski, "Examining sustainability indicators of space management in elderly Facilities-a case study in China," Journal of Cleaner Production, vol. 208, pp. 144-159, Jan. 2019.
5. M. Ahmad, I. Ahmed, and A. Adnan, "Overhead View Person Detection Using YOLO," 2019 IEEE 10th Annual Ubiquitous Computing, Electronics & Mobile Communication Conference (UEMCON), pp. 0627-0633, Oct. 2019.
6. ASHRAE, Ventilation and Acceptable Indoor Air Quality, ANSI/ASHRAE Standard 62.1-2022, Atlanta, GA, USA, 2022.
7. 內政部建築研究所,《智慧建築評估手冊》(2024年版), 台北, 台灣: 內政部建築研究所, 2024.
8. S. Lavy, J. A. Garcia, and M. K. Dixit, "Establishment of KPIs for facility performance measurement: review of literature," Facilities, vol. 28, no. 13/14, pp. 440-464, 2010.
9. E. T. Hall, The Hidden Dimension. New York, NY, USA: Doubleday, 1966.
10. J. Gehl, Life Between Buildings: Using Public Space. Washington, DC, USA: Island Press, 2011.
11. A. Farr, T. Kleinschmidt, P. Yarlagadda, and K. Mengersen, "Wayfinding: A simple concept, a complex process," Transport Reviews, vol. 32, no. 6, pp. 715-743, 2012.
12. K. Lynch, The Image of the City. Cambridge, MA, USA: MIT Press, 1960.
