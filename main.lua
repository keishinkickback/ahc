
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
--  TEST
-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

torch.setdefaulttensortype('torch.FloatTensor')
agg = require 'aggclustering'

local x = torch.Tensor({ { 1,  1},
                   { 2,  1},
                   {-1, -1},
                   {-1, -2},
                   {-2, -2} })

print(x)
local groups, dists, md = agg:clustering_threshold(x, agg.Cosine, 0.2, 'single')
print(groups, dists, md)
local groups, dists, md = agg:clustering_nClusters(x, agg.Euclidean, 2, 'complete')
print(groups, dists, md)
