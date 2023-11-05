from anytree import Node, RenderTree

# transactions
transactions = [
    ["Bread", "Milk"],
    ["Bread", "Diapers", "Beer", "Eggs"],
    ["Milk", "Diapers", "Beer", "Cola"],
    ["Bread", "Milk", "Diapers", "Beer"],
    ["Bread", "Milk", "Diapers", "Cola"],
    ["Bread", "Beer", "Cola"],
    ["Bread", "Milk", "Eggs"],
    ["Bread", "Milk"],
    ["Bread", "Eggs"],
    ["Diapers", "Beer", "Cola"],
    ["Milk", "Diapers", "Beer", "Cola"],
    ["Bread", "Milk", "Diapers", "Beer", "Eggs", "Cola"]
]

# Initialize a dictionary to count occurrences
element_counts = {}

# Iterate through transactions and count occurrences of items
for transaction in transactions:
    for item in transaction:
        element_counts[item] = element_counts.get(item, 0) + 1

# Frequency of each individual item
print(element_counts)

# Sort transactions based on the sum of item occurrences
sorted_transactions = sorted(transactions, key=lambda transaction: sum(element_counts[item] for item in transaction), reverse=True)

# Ordered-item set
print(sorted_transactions)

# Create the root of the tree
root = Node("Root", count=1)

# Iterate through frequent patterns and add them to the tree
for itemset in sorted_transactions:
    current_node = root
    for item in itemset:
        # Check if the node exists, if not, create it
        child_node = next((child for child in current_node.children if child.name == item), None)
        if child_node is None:
            child_node = Node(item, parent=current_node, count=1)
        else:
            current_node.count = current_node.count + 1
        current_node = child_node

# Display the tree
for pre, fill, node in RenderTree(root):
    print(f"{pre}{node.name}({node.count})")
