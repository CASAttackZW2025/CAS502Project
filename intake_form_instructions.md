# CAS502Project Intake Form Instructions

Welcome to the CAS502Project intake form! This document provides detailed instructions for completing your production system intake form (e.g., production_system_intake_Example01.xlsx). Accurate and complete data is essential for the simulation to run correctly and produce meaningful results. We all know the saying garbage in, garbage out; if the production intake data is imputed incorrectly, the result will be garbage. Please read these instructions carefully and refer to the example file provided for guidance.

## General Guidelines
Please complete every section! Each sheet in the intake form must be filled out with data for your given production system. Take time add fill out all the tables you can. If the table is not relevant, leave it blank. Use consistent units and formats across all sheets (e.g., use integers for indices and numerical values, proper date formats, etc.). Double-check all entries to avoid errors, and remember that inaccurate or missing data may cause the simulation to fail or produce incorrect results.
Please refer to examples as you fill ut the data. The sample intake form (production_system_intake_Example01.xlsx) shows how a correctly completed form should look.

## Sheet-by-Sheet Instructions:

### ProcessorTypeTbl
Purpose: Define the different processor types used in your production system.  
Required Fields:
ProcessorTypeIndex: A unique integer (e.g., 0, 1, 2, …).  
ProcessorTypeID An alphanumeric identifier (e.g., "PTYPE01").  
ProcessorType A descriptive name (e.g., "NORMAL_PROCESSOR" or "ACQUIRE_SCHD_SOURCE").

### LocationTbl 
Purpose: Describe the physical or logical locations where production takes place.  
Required Fields:
LocationIndex: A unique integer identifier  (e.g., 0, 1, 2, …).  
LocationID: A short alphanumeric code (e.g., "LOC01").  
LocationName: The name of the location (e.g., "Main Facility").  
AreaDesc: A brief description of the area (e.g., “Assembly Area”). 
SQFT: The size of the area in square feet. 

### ProcessorTbl  
Purpose: List individual processors and link them to their type and location.  
Required Fields:
ProcessorIndex, ProcessorID, ProcessorDesc: Basic identification and description.  
ProcessorTypeIndex, ProcessorTypeID, ProcessorType: Link to the processor type defined in ProcessorTypeTbl.  
NumberStations: The number of stations associated with the processor.  
LocationIndex, LocationID, Location: Link to a location in LocationTbl.  
DownProabilityPerUnity, DownDelay: Downtime probability and delay for the processor.  
ProcessingUnits: Information about processing units used.

### StationTbl
Purpose: Describe work stations attached to each processor.  
Required Fields: 
StationIndex, StationID, StationDesc: Unique identifier and description of the station.  
StationType: The type of station (e.g., "REWORK").  
StationPriorityMethod: The method for prioritizing tasks (e.g., "FIFO").  
StationCapacity: Maximum number of items or workers the station can handle.  
ProcessorIndex, ProcessorID, ProcessorDesc: Link the station to its processor.  
DownProabilityPerUnit, DownDelay: Downtime details for the station.  
Units: Measurement units (e.g., "items", "workers").

### TaskTbl
Purpose: List the production tasks to be performed.  
Required Fields: 
TaskIndex Unique task identifier (integer).  
TaskID An alphanumeric code for the task.  
TasksDesc A brief description of the task.  
ProcessorIndex/ProcessorID/ProcessorDesc: Link the task to a specific processor.  
GenRateProcessTime: This is the processing time used if the standard rate and the variable rate are not know.
StndrdRateProcessTime: This is the mechanically minimum time if a person were a machine completing the work, this is the floor of the learning curve that it will never hit. 
VarRateProcessTime: This is time that it takes at the start of manufacturing above the standard rate. This is what the learning curve applies too.
LearnCurvePct: The Wright Patterns learning curve models the percent of the time the next unit takes after the first unit. Each units production time gets reduced. 
LearnCurveMaxThreshold: The maximum threshold of the first unit that subsequent units can be reduced.
MinProcessTimePct: For a PERT or triangular distribution of processing time.
MaxProcessTimePct: For a PERT or triangular distribution of processing time.
YeildPct: Likelihood of product that is yielded without rework.
Batching Details: BatchSizeReq lets you know how many units are needed to make a single batch unit, and UnbatchingSize tells how many units are separated.
ComponentsGenerated: List the components produced by the task.

### TaskResourcesTbl
Purpose: Specify the resource requirements for each task.  
Required Fields:
TaskIndex, TaskID, TasksDesc: These should match the corresponding task in TaskTbl.  
Resource Columns (NResource01 to NResourceN): Enter the required quantity for each resource. If a resource is not needed, you may leave its field blank.

### ResourceTbl
Purpose: Detail each available resource in your production system.  
Required Fields:
ResourceIndex: A unique integer identifier for the resource.  
ResourceIndex.1: An alternative or secondary identifier (if applicable).  
ResourceName: The name of the resource.  
ResourceType: The type or category (e.g., "EGSE").  
TotalResourceUnitsAvailable: Total available units of the resource.  
DownProbabilityPerUnit: Probability (as an integer or percentage) that a unit is unavailable.  
DownReplacementDelay: Delay (in time units) before a down unit is replaced.  
  Units: The unit of measurement (e.g., "hours").

### ProductsTbl
Purpose: Identify products produced by the system.  
Required Fields:  
ProductIndex, ProductID, ProductDesc: Basic product identification and description.

### ConfigTbl  
Purpose: Define configurations that link products.  
Required Fields:
ConfigIndex, ConfigID, ConfigDesc: Identification and description of the configuration.  
ProductIndex, ProductID, ProductDesc: Link the configuration to a product.

### ProcessUnitTbl 
Purpose: Detail individual processing units (or work cells) within a configuration.  
Required Fields:
ProcessUnitIndex, ProcessUnitID, ProcessUnitDesc: Identification of the processing unit.  
ConfigIndex, ConfigID, ConfigDesc: Link to the configuration.  
ProductIndex, ProductID, ProductDesc: Link to the product.

### ArrivalUnitTbl
Purpose: Define the units that represent incoming work or materials.  
Required Fields:
ArrivalUnitIndex, ArrivalUnitID, ArrivalUnitDesc: Identification and description.  
ConfigIndex, ConfigID, ConfigDesc: Configuration details.  
ProductIndex, ProductID, ProductDesc: Product linkage.

### ComponentsTbl  
Purpose: List components produced or required in the production process.  
Required Fields:
ComponentIndex, ComponentID, ComponentDesc: Basic component details.  
UnitType, UnitIndex, UnitID, UnitDesc: Details about the component's unit.  
ConfigIndex, ConfigID, ConfigDesc, ProductIndex, ProductID, ProductDesc: Link to configuration and product.

### ProductN
Purpose: Provide task-specific data for ProductN.  
Required Fields: 
TaskPred: A boolean of a task predicate 
Task 1 to Task Last task N: Columns representing detailed task information. Column headers for tasks may be auto-renamed; verify the final names.

### ArrivalDatesTbl (only If applicable. You will need need either dates or rates or both)
Purpose: Specify the scheduled arrival dates for materials or work.  
Required Fields:
ArrivalDatesIndex, ArrivalDatesID, ArrivalDatesDesc: Basic date record details.  
ArrivalDate: The scheduled date/time of arrival (use proper datetime format).  
ArrivalCount: Number of arrivals on that date.  
ArrivalUnitIndex, ArrivalUnitID, ArrivalUnitDesc: Link to the arrival unit.  
ProductIndex, ProductID, ProductDesc: Product linkage.

### ArrivalRatesTbl (only If applicable. You will need either dates or rates or both)
Purpose: Provide the arrival rate of materials or work items.  
Required Fields:
ArrivalRatesIndex, ArrivalRatesID, ArrivalRatesDesc: Identification details.  
ArrivalValue: Numeric arrival rate (e.g., items per hour).  
ArrivalUnits: Units for the rate (e.g., "per hour").  
ArrivalCount: How many items are arriving at that rate.  
ArrivalUnitIndex, ArrivalUnitID, ArrivalUnitDesc: Arrival unit linkage.  
ProductIndex, ProductID, ProductDesc: Product linkage.

### DeliveryRatesTbl (only If applicable. You will need either dates or rates or both)
Purpose: List the delivery rates (how quickly products are delivered).  - Required Fields:
DeliveryRatesIndex, DeliveryRatesID, DeliveryRatesDesc: Identification details.  
DeliveryRateValue: Numeric value for delivery speed.  
DeliveryUnits: Units (e.g., “per hour”).  
DeliveryCount: Count of deliveries at that rate.  
ProductIndex, ProductID, ProductDesc: Link to the product delivered.

### DeliveryDatesTbl (only If applicable. You will need either dates or rates or both)  
Purpose: Specify scheduled product delivery dates.  
Required Fields:  
DeliveryRatesIndex, DeliveryRatesID, DeliveryRatesDesc: Identification linking back to delivery rates.  
DeliveryRateDate: The scheduled delivery date/time.  
DeliveryCount: Number of deliveries on that date.  
ProductIndex, ProductID, ProductDesc: Product linkage.

### PersonnelTbl
Purpose: Capture details about the workforce assigned to production.  
Required Fields:  
PersonnelIndex, PersonnelID, PersonnelDesc: Identification of personnel or groups.  
Npersonnel: Number of personnel.  
ProductionRate: Rate at which personnel produce.  
HourlyRate: Cost per hour for the personnel.  
Capacity: Maximum workload capacity.  
Schedual, Shift: Scheduling information.  
HolidayCalendar, HolidayDays: Information on holidays.  
PTOUtilization, PTORatePerWeek, PTOCalendar: Paid time off details.  
OnTimeStartReliability, OnTimeStartFailureDelay: Shift start performance metrics.  
OvertimeAllowance, OvertimeRate: Overtime details.

### SchdPtrnTbl
Purpose: Define weekly scheduling patterns (shift codes per day).  
Required Fields:
SchdPtrnIndex, SchdPtrnID, SchdPtrnDesc: Identification of the schedule pattern.  
ShiftStartDate: Start date for the schedule pattern.  
DayN – DayN: Codes or descriptions for each day in the model in a repeating pattern.

### ShiftTbl
Purpose: Detail each work shift's structure and timing.  
Required Fields:  
ShiftIndex, ShiftID, ShiftDesc: Identification and description of the shift.  
ShiftStart, ShiftEnd: Start and end times.  
ShiftStartUnit, ShiftEndUnit: Numeric representations of times (if needed).  
ShiftLength: Total duration of the shift.  
ShiftBreakStart, ShiftBreakEnd: Timing of break periods.  
ShiftBreakStartUnit, ShiftBreakEndUnit, BreakLength: Break details.  
StartBuffer, EndBuffer: Buffers before and after the shift.  
ProductionStartTimePart1/Part2, ProductionEndTimePart1/Part2: Segments of production time.  
TotalStaffingTime, BaseProductionTime: Overall staffing and baseline production time.

### HolidayClndrTbl
Purpose Provide a calendar of holidays affecting production.  
Required Fields
YearN: The calendar year.  
HolidayCalendarN: List of holiday dates (in datetime format).  
HolidayCalendarDayN: An identifier or day count for each holiday.

### PTOTbl
Purpose: Capture details of paid time off (PTO) for personnel.  
Required Fields: 
PTOYearN: The year for the PTO record.  
PTOCalendarN: List of PTO dates (datetime format).  
PTODayN, PTODayN: Specific PTO day details (e.g., start and end dates or durations).  
PTOYearlySumN: Total PTO days allowed or used.

## Data Formatting and Best Practices
Consistency: Enter data consistently. For example, use the same date format (YYYY-MM-DD) and ensure numeric fields contain only numbers.  
Units: Verify that measurement units (e.g., hours, square feet) are consistent across sheets.  
Completeness: Only leave fields blank if they are not applicable to your system.  
Review: After completing each sheet, double-check your entries against this guide and the sample intake form.

## Instructions
Save Your Form: Once completed, save your intake form as an Excel file (.xlsx).
Upload to Google Colab:  When running the simulation in the Google Colab notebook, use the provided file upload widget to select and upload your completed intake form.
Troubleshooting:  If you encounter issues during the upload or simulation, refer to the project’s documentation or submit an issue on GitHub for assistance.

## Additional Resources
Sample Intake Form: Review production_system_intake_Example01.xlsx for an example of a correctly completed intake form.
Excel Data Formatting: Consult online resources for guidance on proper Excel formatting if needed.


Thank you for carefully completing your intake form!
