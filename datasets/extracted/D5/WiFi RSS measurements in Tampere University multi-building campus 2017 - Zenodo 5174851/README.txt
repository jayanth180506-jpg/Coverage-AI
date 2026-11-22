This package contains two Wi-Fi RSSI Datasets, namely TIE1 and SAH1.

---TIE1 was collected in the Tietotalo building from Tampere
   University during the period of August-December 2017 and it is
   composed of four files:
   ---TIE1_training_rss.csv Radio map or reference dataset as a matrix
      with 10633 reference samples (rows) and 613 APs (columns) 
   ---TIE1_training_coordinates.csv Ground truth location of the 10633
      reference samples (rows) as X,Y,Z and floor being the
      coordinates in a local system
   ---TIE1_test_rss.csv Operational fingerprints as a matrix with 50
      evaluation samples (rows) and 613 APs (columns) 
   ---TIE1_test_coordinates.csv Ground truth location of the 50
      evaluation samples (rows) as X,Y,Z and floor being the
      coordinates in a local system

---SAH1 was collected in the Sahkotalo building from Tampere
   University during the period of October-December 2017 and it is
   composed of four files:
   ---SAH1_training_rss.csv Radio map or reference dataset as a matrix
      with 9291 reference samples (rows) and 775 APs (columns) 
   ---SAH1_training_coordinates.csv Ground truth location of the 9291
      reference samples (rows) as X,Y,Z and floor being the
      coordinates in a local system
   ---SAH1_test_rss.csv Operational fingerprints as a matrix with 156
      evaluation samples (rows) and 775 APs (columns) 
   ---SAH1_test_coordinates.csv Ground truth location of the 156
      evaluation samples (rows) as X,Y,Z and floor being the
      coordinates in a local system

Additional considerations:

 - All the RSSI values are stored as are they were registered with a
   regular smartphone. In order to generate the fingerprint vectors,
   the non-detected APs where filled with the default bogus
   value +100dBm. i.e., the value +100dBm in the training and test
   rss files indicates that the AP was not detected. Generally, this
   default bogus value has been replaced with a low RSSI value in
   experiments, such as -100dBm, -110dBm, -150dBm or -200dBm.

 - The coordinates X,Y,Z are expressed in meters in a local coordinate
   system. Additionally, the sequential floor identifier is also
   provided.

Please cite the source at Zenodo when using any of these two datasets:
      Lohan, E.S; Torres-Sospedra, J. and Gonzalez, A.; 
      "WiFi RSS measurements in Tampere University multi-building
      campus, 2017" Zenodo, 2021. 
      https://zenodo.org/record/5174851
      https://www.doi.org/10.5281/zenodo.5174851