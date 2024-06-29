
## Dataset

This dataset contains information about forest fires in Algeria, collected between June 2012 and September 2012, focusing on two distinct regions:

* **Bejaia region:** Located in the northeast of Algeria.
* **Sidi Bel-abbes region:** Located in the northwest of Algeria.

### Key Features

* **Instances:** 244 (122 for each region).
* **Attributes:** 11 attributes (features).
* **Output Attribute:** 1 output attribute (class).
* **Classes:**
    * **Fire:** 138 instances.
    * **Not fire:** 106 instances.

### Attributes Description

| Feature           | Description                                          | Data Type |
|--------------------|---------------------------------------------------|-----------|
| **Classes**        | Fire or not fire (target variable)                 | Categorical |
| **month**         | Month of the year (1-12)                            | Integer   |
| **RH**            | Relative humidity (%)                               | Integer   |
| **Temperature**    | Temperature (Celsius)                               | Integer   |
| **Ws**            | Wind speed (km/h)                                   | Integer   |
| **year**          | Year of the observation                              | Integer   |
| **DC**            | Drought Code Index                                    | Float     |
| **Rain**          | Total amount of precipitation (mm)                   | Float     |
| **DMC**           | Drought Code Index                                    | Float     |
| **FFMC**          | Fine Fuel Moisture Code                              | Float     |
| **BUI**           | Buildup Index                                       | Float     |
| **ISI**           | Initial Spread Index                                | Float     |
| **FWI**           | Fire Weather Index                                   | Float     |
| **day**           | Day of the month (1-31)                             | Integer   |
