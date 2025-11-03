def get_knowledge_slices_prompt(document, topic):
    prompt = "You will receive a document and a topic. Your task is to identify the knowledge-slices within the document that are very relevant to the " \
    "given topic. A knowledge-slice is a piece of information representing the highlights of the document related to the given topic i.e. each knowledge-slice should be such that it both " \
    "represents an important point in the document, but at the same time, the knowledge-slice should pertain closely to the given topic. Also, the knowledge-slice " \
    "should not represnt any additional information that is not present in the document." + "\n\n[Document]\n" + document + "\n\n[Topic]\n" + topic + "\n\nPlease "
    "ONLY return the relevant knowledge-slices in the form of a list enclosed within square brackets. Your response should be in the "
    "following format:\n[Knowledge-Slices]\n[Knowledge-Slice 1, Knowledge-Slice 2,..., Knowledge-Slice n]\n\n Your response:"
    return prompt


def build_subset_topic_prompt(taxonomy_topic, documents):
    prompt = "A taxonomy is a tree-structured semantic hierarchy that establishes a classification of the existing literature under a common topic. " \
    "You will receive a taxonomy topic along with a collection of document highlights. Your task is to find a sub-topic that can act as a child node to the " \
    "taxonomy topic. The sub-topic should be representative of all the given document highlights. The sub-topic should be a short phrase. \n\n[Taxonomy " \
    "Topic]\n" + taxonomy_topic + "\n\n[Documents]\n\n" + "".join(documents) + "\n\nPlease ONLY return the sub-topic. \n[Your response]"
    return prompt


def build_skeleton_prompt(taxonomy_topic, documents):
    prompt = "A taxonomy is a tree-structured semantic hierarchy that establishes a classification of the existing literature under a common topic. " \
    "You will receive a taxonomy topic along with a collection of documents. Your task is to create a taxonomy tree " \
    "using the given topic and based on the highlights of the documents i.e. create new child nodes by identifying " \
    "generalizable sub-level topics from the document highlights that can act as child nodes to the taxonomy topic, " \
    "which acts as the root node. The taxonomy tree should be created such that it looks as if all the given documents " \
    "are a part of the taxonomy. There may be several levels in the tree i.e. each node " \
    "may contain child nodes, but the total depth of the tree should not exceed three. The topics in all the levels of the tree except " \
    "the last level must not be too specific so that it can accommodate future sub-topics i.e. child nodes.\n    - The nodes at the last level of the " \
    "hierarchy i.e. the leaf nodes should reflect a single topic instead of a combination of topics.\n    - Each node label is a small and concise " \
    "phrase.\n\n[Response Format Instructions]\n    - The output tree is to be formatted as shown in the example such that the root node is the taxonomy " \
    "topic and each child node is connected to its parent.\n\n[Example Output]\n\n Security and Privacy on Blockchain\n|-- CONSENSUS ALGORITHMS\n|   " \
    "|-- BFTbased Consensus Algorithms\n|   |-- Comparison of Consensus Algorithms\n|   |-- Other Consensus Algorithms\n|   |-- Proof of State PoS\n|   " \
    "+-- Proof of Work PoW\n|-- PRIVACY AND SECURITY TECHNIQUES USED IN BLOCKCHAIN\n|   |-- Anonymous Signatures\n|   |-- AttributeBased Encryption ABE\n|   " \
    "|-- Game-based Smart Contracts\n|   |-- Homomorphic Encryption HE\n|   |-- Mixing\n|   |-- Non-Interactive Zero-Knowledge NIZK Proof\n|   " \
    "+-- Secure Multi-Party Computation\n+-- SECURITY AND PRIVACY PROPERTIES OF BLOCKCHAIN\n    +-- Basic Security Properties. \n\n[Taxonomy " \
    "Topic]\n" + taxonomy_topic + "\n\n[Documents]\n\n" + "".join(documents) + "\n\nPlease ONLY return the taxonomy tree"
    " in the output format as shown in the example above. \n[Your response]"
    return prompt

"""def build_subset_skeleton_prompt(taxonomy_topic, documents):
    prompt = "A taxonomy is a tree-structured semantic hierarchy that establishes a classification of the existing literature under a common topic. " \
    "You will receive a taxonomy topic along with a group of document collections. Your task is to create a taxonomy tree " \
    "using the given topic and based on the highlights of the documents i.e. create new child nodes by identifying " \
    "generalizable sub-level topics from the document highlights that can act as child nodes to the taxonomy topic, which acts as the root node " \
    ". The taxonomy tree should be created such that it looks as if all the given documents are a part of the taxonomy. There may be several levels in the tree i.e. each node " \
    "may contain child nodes, but the total depth of the tree should not exceed three. " \
    "Each document collection is given separately as input and each document collection represents a separate "\
    "sub-topic (child node) of the taxonomy topic. The topics in all the levels of the tree except " \
    "the last level must not be too specific so that it can accommodate future sub-topics i.e. child nodes.\n    - The nodes at the last level of the " \
    "hierarchy i.e. the leaf nodes should reflect a single topic instead of a combination of topics.\n    - Each node label is a small and concise " \
    "phrase.\n\n[Response Format Instructions]\n    - The output tree is to be formatted as shown in the example such that the root node is the taxonomy " \
    "topic and each child node is connected to its parent.\n\n[Example Output]\n\nEnergy Efficient Computing Systems: Architectures," \
    " Abstractions and  Modeling to Techniques and Standards\n|-- MICROARCHITECTURAL TECHNIQUES\n|   |-- Microarchitectural techniques for AI accelerators\n|" \
    "   |-- Microarchitectural techniques for CPUs\n|   |-- Microarchitectural techniques for GPUs\n|   +-- Microarchitectural techniques for Memory\n|-- " \
    "MODELING AND SIMULATION\n|   |-- Accelerator Simulators\n|   |-- Cache Simulators\n|   |-- GPU Simulation\n|   |-- Memory Simulators\n|   " \
    "|-- Processor and full system Simulators\n|   +-- Thermal Modeling\n|-- SPECIFICATION\n|   |-- ACPI and DeviceTree\n|   |-- Accellera Unified " \
    "Power Format\n|   +-- IEEE 1801 Unified Power Format\n|-- SYSTEM LEVEL TECHNIQUES FOR ENERGY EFFICIENCY\n|   |-- Energy Efficiency in AMD " \
    "ProcessorsSOCs\n|   |-- Energy Efficiency in Intel ProcessorsSOCs\n|   |-- Intel x86 Power Management\n|   |-- OS and Software Techniques\n|" \
    "   +-- The rise of ARM in enterprise HPC and the Cloud\n|-- THE ROAD AHEAD AND NEW TRENDS\n+-- VERIFICATION\n    |-- Benchmarks\n    " \
    "|-- Consortiums\n    |-- Cross Layer Optimizations for Energy Efficiency\n    +-- Standards \n\n[Taxonomy " \
    "Topic]\n" + taxonomy_topic + "\n\n[Documents]\n\n" + "\n\n".join(documents) + "\n\nPlease ONLY return the taxonomy tree"
    " in the output format as shown in the example above. \n[Your response]"
    return prompt
"""

def build_skeleton_prompt_without_docs(taxonomy_topic):
    prompt = "A taxonomy is a tree-structured semantic hierarchy that establishes a classification of the existing literature under a common topic. " \
    "You will receive a taxonomy topic. Your task is to create a taxonomy tree " \
    "using the given topic i.e. create new child nodes by identifying " \
    "generalizable sub-level topics that can act as child nodes to the taxonomy topic, which acts as the root node " \
    ".  There may be several levels in the tree i.e. each node " \
    "may contain child nodes, but the total depth of the tree should not exceed three. The topics in all the levels of the tree except " \
    "the last level must not be too specific so that it can accommodate future sub-topics i.e. child nodes.\n    - The nodes at the last level of the " \
    "hierarchy i.e. the leaf nodes should reflect a single topic instead of a combination of topics.\n    - Each node label is a small and concise " \
    "phrase.\n\n[Response Format Instructions]\n    - The output tree is to be formatted as shown in the example such that the root node is the taxonomy " \
    "topic and each child node is connected to its parent.\n\n[Example Output]\n\nSecurity and Privacy on Blockchain\n|-- CONSENSUS ALGORITHMS\n|   " \
    "|-- BFTbased Consensus Algorithms\n|   |-- Comparison of Consensus Algorithms\n|   |-- Other Consensus Algorithms\n|   |-- Proof of State PoS\n|   " \
    "+-- Proof of Work PoW\n|-- PRIVACY AND SECURITY TECHNIQUES USED IN BLOCKCHAIN\n|   |-- Anonymous Signatures\n|   |-- AttributeBased Encryption ABE\n|   " \
    "|-- Game-based Smart Contracts\n|   |-- Homomorphic Encryption HE\n|   |-- Mixing\n|   |-- Non-Interactive Zero-Knowledge NIZK Proof\n|   " \
    "+-- Secure Multi-Party Computation\n+-- SECURITY AND PRIVACY PROPERTIES OF BLOCKCHAIN\n    +-- Basic Security Properties. \n\n[Taxonomy " \
    "Topic]\n" + taxonomy_topic + "\n\nPlease ONLY return the taxonomy tree"
    " in the output format as shown in the example above. \n[Your response]"
    return prompt
