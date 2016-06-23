using SyntheticImplicitFeedback

# define a set of features
features = Feature[]

push!(features, Feature("Age", 1950, () -> rand(1930:2000)))
push!(features, Feature("Sex", 0, () -> rand(0:1)))
push!(features, Feature("Geo", 1, () -> rand(1:50))) # US has 50 states
push!(features, Feature("Ad", 0, () -> rand(0:4)))

# define a set of rules
rules = Rule[]

push!(rules, Rule(s -> true, 0.001))
push!(rules, Rule(s -> s["Ad"] == 2, 0.01))
push!(rules, Rule(s -> s["Age"] >= 1980 && s["Age"] <= 1989 && s["Geo"] == 32 && s["Ad"] == 0, 0.30)) # New York
push!(rules, Rule(s -> s["Age"] >= 1950 && s["Age"] <= 1959 && s["Geo"] == 32 && s["Ad"] == 1, 0.30))
push!(rules, Rule(s -> s["Age"] >= 1980 && s["Age"] <= 1989 && s["Geo"] == 3 && s["Ad"] == 1, 0.30)) # Arizona
push!(rules, Rule(s -> s["Age"] >= 1950 && s["Age"] <= 1959 && s["Geo"] == 3 && s["Ad"] == 0, 0.30))

function run(features, rules, N)
    for i in 1:N
        s = Dict()

        for f in features
            s[f.name] = f.random()
        end

        generate(s, rules) && println(join(values(s), "\t"))
    end
end

# first generate 0.5 million impressions
run(features, rules, 500000)

# change the most popular ad from #2 to #3
rules[2].f = (s -> s["Ad"] == 3)

# generate additional 0.5 million impressions
run(features, rules, 500000)
