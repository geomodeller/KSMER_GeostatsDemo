# Geostatistics Demonstration - SGS/SIS/MPS

### Honggeun Jo, Assistant Professor, Inha University (Korea)
This demo showcases an implementation of basic geostatistical modelling, including sequential Gaussian simulation (**SGS**), sequential indicator simulation (**SIS**), and multi-point statistics (**MPS**). In this demo, we first import 2D well information with various reservoir properties, such as porosity, acoustic impedence, facies, permeability and brittleness. Then we calculate variogram to assess spatial continuity in the reservoir. Finally, with the designed variogram, we run SGS with (1) Python, which will be readily too heave as dimension rises, and (2) GSLIB, programed in Fortran by Dr. Clayton Deuatsch.:

1. Load open packages and import the CSV file (regarding well data) into Python using Pandas.
2. Visualize the well data.
3. Compute variogram.
4. **Implement SGS using Python**
5. **Repeat SGS with GSLIB**
6. **run SIS (for categorical facies) with GSLIB**
7. **run MPS (for categorical facies) with GSLIB**

The demo is presented by Honggeun Jo, an Assistant Professor at Inha University (Korea). You can reach out to him through his contacts on [Youtube/whghdrms](https://www.youtube.com/@whghdrms) | [GitHub/whghdrms](https://github.com/whghdrms) |  [GoogleScholar](https://scholar.google.com/citations?user=u0OE5CIAAAAJ&hl=en) | [LinkedIn](https://www.linkedin.com/in/honggeun-jo/)

**Note that this workflow was originally developed by Dr. Pyrcz, and the details can be found from the https://github.com/GeostatsGuy/GeostatsPy **
