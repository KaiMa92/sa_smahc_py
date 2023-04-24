| **name**            | **format** | **description** |
|---------------------|------------|-----------------|
| Actuator type|SMAHC object| The type of actuator to simulate given in the Material library|
| Ambient temperature |float           |Initial temperature of the SMAHC system and ambient temperature|
| Sequences           |list            |List containing the sequences to simulate given as list [Ampere, Time]|
| Time increment      |float            |The time increment in seconds           |
| Spatial increment   |float            |The spatial increment for the temperature field in meter|
| mf0                 |float            |The initial martensite fraction|
| stress0             |float            |The initial stress|
| Alpha elastomer     |float            |The heat transfer coefficient between interlayer and surroundings|
| Load                |float            |External point load pointing in the negative z-direction at the edge of the active area |
| data_resolution     |int            |Every nth datapoint will be stored|