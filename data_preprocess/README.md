## Environment
joern 0.3.1, neo4j 2.1.5, py2neo 2021.2.3, python 3.6 
## Generating Graph-based Vulnerability Candidate slices (i.e., GrVCs)

We use joern to parse source code. The input is source code files, and the output is a file named .joernIndex.
```
java -jar .../joern-0.3.1/bin/joern.jar .../source_code/Linux/
```
Next, we start the Neo4j database to enable Python to query the code structure graphs stored within it.
 ```
cd /usr/lib/neo4j-community-2.1.5/bin
./neo4j start-no-wait
```  
1. Generating Control Flow Graph (CFG)
```
python get_cfg_relation.py
```
2. Generating Program Dependency Graph (PDG)
```
python complete_PDG.py
```
3. Constructing the Call Graph (CG) of functions
```
python access_db_operate.py
```
4. Extracting Slicing Entry Nodes
```
python entry_nodes_get.py
```
5. Extracting GrVCs
```
python extract_GrVCs.py
```

