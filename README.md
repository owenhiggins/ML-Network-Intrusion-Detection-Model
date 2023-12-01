# ML Network Intrusion Detection Model
Using machine learning with Netflow data, to detect anomalous events on a network. Thus proving the feasibility to prevent zero day attacks with Network Intrusion Dection Systems combined with machine learning.
Using a data pipleine with AWS, Docker, Airflow, Tensorflow and more.
## Description

### Objective
Network Intrusion Detection Systems are used to monitor networks for malicious activity and prevent network or data breaches. There is one drawback with these systems; Network Intrusion Detection Systems fail to protect against Zero-Day exploits, which are never before seen attacks used by threat actors. With the vastly growing threat landscape in cyberspace, it is becoming more and more important to be able to protect and defend against these types of attacks. Using machine learning, we plan to be able to detect anomalies on Network Intrusion Detection Systems. A machine learning approach to this issue will allow for rapid and continuous improvement to the model as new threats emerge and are documented. We will use Tableau to visualize what attacks we would have been able to prevent using our machine learning model. A mix of tabular data like confusion matrices and visualizations like stacked bar or column charts should be able to convey the results to all audiences. 

### Dataset
This dataset contains NetFlow packet capture data labeled as benign or attack, and includes 12 features and contains around 12 million records. Of these records 77% are benign and 23% are attacks. This dataset can be used to train machine learning based network intrusion detection systems. The NetFlow format is commonly used on networks which allows for wide deployment and scaling of machine learning based network intrusion detection systems.
- Features:
- IPv4 source address
- IPv4 destination address
- IPv4 source port number
- IPv4 destination port number
- IP protocol identifier byte
- Cumulative of all TCP flags
- Layer 7 protocol (numeric)
- Incoming number of bytes
- Outgoing number of bytes
- Incoming number of packets
- Outgoing number of packets
- Flow duration in milliseconds

### 
Dataset - [Machine Learning-Based NIDS Datasets](https://staff.itee.uq.edu.au/marius/NIDS_datasets/#RA5) 

Paper - [NetFlow Datasets for Machine Learning-Based Network Intrusion Detection Systems](https://doi.org/10.1007/978-3-030-72802-1_9)

### Tools & Technologies

- Cloud - [**AWS**](https://aws.amazon.com/)
- Containerization - [**Docker**](https://www.docker.com)
- Orchestration - [**Airflow**](https://airflow.apache.org)
- Data Lake - [**AWS S3**](https://aws.amazon.com/s3/)
- Data Warehouse - [**AWS S3**](https://aws.amazon.com/s3/)
- Language - [**Python**](https://www.python.org)

### Architecture
![Diagram.png](https://github.com/owenhiggins/Zero-Day-Exploit-Prediction-through-Anomaly-Detection-for-Network-Intrusion-Detection-Systems/blob/main/Diagram.png)

### Contact Information
- Owen Higgins owenhiggins@vt.edu
- Casey Gary *insert email*
