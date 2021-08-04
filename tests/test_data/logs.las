~VERSION INFORMATION
 VERS.                          2.0 :   CWLS LOG ASCII STANDARD -VERSION 2.0
 WRAP.                          NO  :   ONE LINE PER DEPTH STEP
~WELL INFORMATION
#MNEM.UNIT              DATA                       DESCRIPTION
#----- -----            ----------               -------------------------
STRT    .M              1000.0000                :START DEPTH
STOP    .M              1010.1500                :STOP DEPTH
STEP    .M              -0.1250                  :STEP
NULL    .               -999.25                  :NULL VALUE
COMP    .               Company_foo              :COMPANY
WELL    .               Well_bar                 :WELL
FLD     .               Field_baz                :FIELD
LOC     .               12-34-12-34W5M           :LOCATION
SRVC    .               ANY LOGGING COMPANY INC. :SERVICE COMPANY
DATE    .               03-MAY-20                :LOG DATE
UWI     .               123456789                :UNIQUE WELL ID
~CURVE INFORMATION
#MNEM.UNIT              API CODES                   CURVE DESCRIPTION
#------------------     ------------              -------------------------
 DEPT   .M                                       :  1  DEPTH
 DT     .US/M           60 520 32 00             :  2  SONIC TRANSIT TIME
 RHOB   .G/CM3           45 350 01 00             :  3  BULK DENSITY
 NPHI   .V/V            42 890 00 00             :  4  NEUTRON POROSITY
 SFLU   .OHMM           07 220 04 00             :  5  SHALLOW RESISTIVITY
 SFLA   .OHMM           07 222 01 00             :  6  SHALLOW RESISTIVITY
 ILM    .OHMM           07 120 44 00             :  7  MEDIUM RESISTIVITY
 ILD    .OHMM           07 120 46 00             :  8  DEEP RESISTIVITY
~PARAMETER INFORMATION
#MNEM.UNIT              VALUE             DESCRIPTION
#--------------     ----------------      -----------------------------------------------
 MUD    .               GEL CHEM        :   MUD TYPE
 BHT    .DEGC           35.5000         :   BOTTOM HOLE TEMPERATURE
 BS     .MM             200.0000        :   BIT SIZE
 FD     .K/M3           1000.0000       :   FLUID DENSITY
 MATR   .               SAND            :   NEUTRON MATRIX
 MDEN   .               2710.0000       :   LOGGING MATRIX DENSITY
 RMF    .OHMM           0.2160          :   MUD FILTRATE RESISTIVITY
 DFD    .K/M3           1525.0000       :   DRILL FLUID DENSITY
~A  DEPTH     DT    RHOB      NPHI   SFLU    SFLA      ILM      ILD
1000.000   123.450 2.550     0.450  123.450  123.450  -999.25  105.600
1010.000   124.000 2.550     0.450  123.450  123.450  -999.25  105.600
1010.150   125.000 -999.25   0.450  123.450  123.450  -999.25  105.600
