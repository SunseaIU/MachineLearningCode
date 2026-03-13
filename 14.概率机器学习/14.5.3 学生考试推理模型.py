from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. 构建模型结构
model = DiscreteBayesianNetwork([
    ('Intelligence', 'Grade'),
    ('Difficulty', 'Grade'),
    ('Intelligence', 'SAT'),
    ('Grade', 'Letter')
])
# 2. 定义 CPDs（条件概率分布）
cpd_difficulty = TabularCPD(variable='Difficulty', variable_card=2,
                            values=[[0.6], [0.4]],
                            state_names={'Difficulty': ['Easy', 'Hard']})

cpd_intelligence = TabularCPD(variable='Intelligence', variable_card=2,
                              values=[[0.7], [0.3]],
                              state_names={'Intelligence': ['Low', 'High']})

cpd_grade = TabularCPD(variable='Grade', variable_card=3,
                       values=[
                           [0.3, 0.05, 0.9, 0.5],
                           [0.4, 0.25, 0.08, 0.3],
                           [0.3, 0.7, 0.02, 0.2]
                       ],
                       evidence=['Intelligence', 'Difficulty'],
                       evidence_card=[2, 2],
                       state_names={
                           'Grade': ['A', 'B', 'C'],
                           'Intelligence': ['Low', 'High'],
                           'Difficulty': ['Easy', 'Hard']
                       })

cpd_sat = TabularCPD(variable='SAT', variable_card=2,
                     values=[[0.95, 0.2],
                             [0.05, 0.8]],
                     evidence=['Intelligence'],
                     evidence_card=[2],
                     state_names={
                         'SAT': ['Low', 'High'],
                         'Intelligence': ['Low', 'High']
                     })

cpd_letter = TabularCPD(variable='Letter', variable_card=2,
                        values=[[0.1, 0.4, 0.99],
                                [0.9, 0.6, 0.01]],
                        evidence=['Grade'],
                        evidence_card=[3],
                        state_names={
                            'Letter': ['Weak', 'Strong'],
                            'Grade': ['A', 'B', 'C']
                        })

# 3. 添加到模型
model.add_cpds(cpd_difficulty, cpd_intelligence, cpd_grade, cpd_sat, cpd_letter)

# 4. 验证模型结构和概率一致性
assert model.check_model()

# 5. 推理1：给定 Grade=B，推断 Intelligence 的概率分布
infer = VariableElimination(model)
result = infer.query(variables=['Intelligence'], evidence={'Grade': 'B'})
print(result)

# 6. 推理2：给定 SAT=High 且 Letter=Strong，推断 Grade 的概率分布
result2 = infer.query(variables=['Grade'], evidence={'SAT': 'High', 'Letter': 'Strong'})
print(result2)