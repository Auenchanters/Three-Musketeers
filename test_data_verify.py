"""Quick test script to verify all data files load correctly."""
from data.generator import (
    load_scenario, load_solution, load_pricing,
    get_available_tasks, get_optimal_savings, get_valid_resize_targets
)

print("=== Data Loader Verification ===\n")

tasks = get_available_tasks()
print(f"Available tasks: {tasks}")

for task_id in tasks:
    scenario = load_scenario(task_id)
    solution = load_solution(task_id)
    
    n_res = len(scenario["resources"])
    n_crit = len(scenario["critical_resources"])
    n_waste = len(scenario["wasteful_resources"])
    total = scenario["_cost_analysis"]["total_monthly"]
    savings = scenario["_cost_analysis"]["optimal_savings_monthly"]
    sol_steps = len(solution["optimal_action_sequence"])
    
    print(f"\n{task_id}:")
    print(f"  Resources: {n_res} ({n_crit} critical, {n_waste} waste)")
    print(f"  Cost: ${total}/mo | Savings: ${savings}/mo")
    print(f"  Solution: {sol_steps} steps")
    if scenario.get("budget_target"):
        print(f"  Budget target: ${scenario['budget_target']}/mo")
    if scenario.get("maintenance_window"):
        print(f"  Maintenance: {scenario['maintenance_window']}")
    if "rightsize_targets" in scenario:
        print(f"  Rightsize targets: {len(scenario['rightsize_targets'])}")
    if "dependency_graph" in scenario:
        print(f"  Dependency edges: {len(scenario['dependency_graph'])}")

pricing = load_pricing()
print(f"\nPricing: {len(pricing['ec2_instances'])} EC2, {len(pricing['rds_instances'])} RDS types")
print(f"Resize m5.xlarge -> {get_valid_resize_targets('m5.xlarge')}")

print("\n=== ALL VERIFIED ===")
