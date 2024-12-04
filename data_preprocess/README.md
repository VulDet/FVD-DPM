## Generating Graph-based Vulnerability Candidate slices (i.e., GrVCs)

1. Use joern to parse source code: the input is source code files, and the output is a file named .joernIndex.
```
java -jar .../joern-0.3.1/bin/joern.jar .../source_code/Linux/
```
2. Start the Neo4j database to enable Python to query the code structure diagrams stored within it.
 ```
cd /usr/lib/neo4j-community-2.1.5/bin
./neo4j start-no-wait
```  
3. get_cfg_relation.py: This file is used to get Control Flow Graph (CFG).

4. complete_PDG.py: This file is used to get Program Dependency Graph (PDG).

5. access_db_operate.py: This file is used to get the Call Graph (CG) of functions.

6. entry_nodes_get.py: This file is used to extract Slicing Entry Nodes. 

7. extract_GrVCs.py: This file is used to extract GrVCs. 

