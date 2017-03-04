require 'math'

local M = {}
local agg = torch.class('agg', M)

-------------------------------------------------------------------------------
--  distance / similarity
-- v1 and v2 are vectors
-------------------------------------------------------------------------------
function agg.Cosine(v1, v2) --> (close) 0 - 1 (far)
  return (1 - (v1 * v2 / ( math.sqrt(v1*v1) * math.sqrt(v2*v2))))/2
end

function agg.Euclidean(v1, v2)
  local vd = torch.csub(v1, v2)
  return math.sqrt(vd * vd)
end

agg.inf = 999 --> max distance

-------------------------------------------------------------------------------
--  private function ( internal use )
-------------------------------------------------------------------------------
function agg:_delete2rows(x, first, second)
  local N = x:size(1)
  if N == 2 then error("invalid input: size needs to be greater than 2.") end
  --> case: 1 sub-matrix
  if first == N - 1 then return x[ { {1, first - 1}, {}}] end
  if second == 2 then return x[ { {second + 1, N}, {}}] end
  if first == 1 and second == N then return x[ { {2, second - 1}, {}}] end
  if first == 1 then --> case: 2 sub-matrices
    return x[ { {2, second - 1}, {}}]:cat( x[ { {second + 1, N},{}}], 1)
  elseif second == N then
    return x[ { {1, first - 1}, {}}]:cat( x[ { {first + 1, second - 1}, {}}], 1)
  elseif first + 1 == second then
    return x[ { {1, first - 1}, {}}]:cat( x[ { {second + 1, N}, {}}], 1)
  else --> case: 3 sub-matrices
    return x[ { {1, first - 1}, {}}]:cat( x[ { {first + 1, second - 1}, {}}], 1)
                                        :cat( x[ { {second + 1, N}, {}}], 1)
  end
end

function agg:_delete2RowsCols(x, first, second)
  return self:_delete2rows(
              self:_delete2rows(x, first, second):transpose(2, 1),first, second)
                    :transpose(2, 1)
end

function agg:_merge2Groups(groups, first, second)
  local tn = {}
  if second < first then first, second = second, first end
  for _, v in ipairs(groups[first]) do table.insert(tn, v) end
  for _, v in ipairs(groups[second]) do table.insert(tn, v) end
  table.remove(groups, first)
  table.remove(groups, second - 1)
  table.insert(groups, tn)
  return groups
end

function agg:_minPoint(dists) --> get indices and max value in distance matrix
  local _, first = torch.min( torch.min(dists, 2), 1)
  local md, second = torch.min( torch.min(dists, 1), 2)
  first, second, md = first[1][1], second[1][1], md[1][1]
  if second  < first then first, second = second, first end
  return first, second, md
end

function agg:_dist2Clusters(x, t1, t2, groups, distf, linkage)
  linkage = linkage or 'single'
  local group1 = groups[t1]
  local group2 = groups[t2]

  if linkage == 'single' then
    local distance = distf(x[group1[1]], x[group2[1]])
    for _, point1 in ipairs(group1) do
      for _, point2 in ipairs(group2) do
        distance = math.min(distance, distf(x[point1], x[point2]))
      end
    end
    return distance
  elseif linkage == 'complete' then
    local distance = distf(x[group1[1]], x[group2[1]])
    for _, point1 in ipairs(group1) do
      for _, point2 in ipairs(group2) do
        distance = math.max(distance, distf(x[point1], x[point2]))
      end
    end
    return distance
  elseif linkage == 'centroid' then
    local c1 = torch.Tensor(x[1]:clone()):zero()
    local c2 = c1:clone()
    for _, point1 in ipairs(group1) do
      c1 = c1 + x[point1]
    end
    for _, point2 in ipairs(group2) do
      c2 = c2 + x[point2]
    end
    c1 = c1 / #group1
    c2 = c2 / #group2
    return distf(c1, c2)
  else
    error("Only 'single', 'complete' and 'centroid' are covered yet.")
  end
end

-------------------------------------------------------------------------------
--  public function
-------------------------------------------------------------------------------
--[[
agg:clustering_threshold(x, distf, threshold, linkage)
agg:clustering_nClusters(x, distf, nClusters, linkage)

        x : matrix, assuming each row represents a vector
    distf : function computing the distance between 2 vectors
            {agg.Cosine, agg.Euclidean}
threshold : threshold to finish clustering
nClusters : #clusters to finish clustering
  linkage : {'single', 'complete', 'centroid'}
]]
function agg:clustering_threshold(x, distf, threshold, linkage)
  linkage = linkage or 'single'
  local N = x:size(1) --> in case all in one cluster
  local groups = {}
  for i = 1, x:size(1) do groups[i] = {i} end --> initial clusters
  local dists = torch.Tensor(x:size(1), x:size(1)):fill(self.inf) --> dists
  for i=1, x:size(1) - 1 do
    for j = i + 1, x:size(1)  do
      dists[i][j] = distf(x[i], x[j])
    end
  end
  local first, second, md = agg:_minPoint(dists) --> max

  while  2 < #groups and md < threshold do
    dists = self:_delete2RowsCols(dists, first, second) --> remove 2 closest
    local n = dists:size(1)
    dists = dists:cat( torch.Tensor(1, n):fill(self.inf), 1)
                                :cat(torch.Tensor( n + 1, 1):fill(self.inf), 2)
    groups = agg:_merge2Groups(groups, first, second)
    for i = 1, dists:size(1) - 1 do --> update dists for only new cluster
      dists[i][dists:size(1)] = agg:_dist2Clusters(x, i, dists:size(1),
                                                        groups, distf, linkage)
    end
    first, second, md = agg:_minPoint(dists) --> max
  end

  if md < threshold then --> if all in one cluster
    local ngroups = {}
    for i = 1, N do table.insert(ngroups,i) end
    return ngroups, torch.Tensor(1, N), threshold
  else
    return groups, dists, md
  end
end

function agg:clustering_nClusters(x, distf, nClusters, linkage)
  linkage = linkage or 'single'
  nClusters = nClusters or 2
  if nClusters < 2 then
    error("invalid input: nClusters needs to be greater than 2.")
  end
  local N = x:size(1) --> in case all in one cluster
  local groups = {}
  for i =1, x:size(1) do groups[i] = {i} end --> initial clusters
  local dists = torch.Tensor(x:size(1), x:size(1)):fill(self.inf) --> dists
  for i = 1, x:size(1) - 1 do
    for j = i + 1, x:size(1) do
      dists[i][j] = distf(x[i], x[j])
    end
  end
  local first, second, md = agg:_minPoint(dists) --> max

  while  nClusters < #groups do
    dists = self:_delete2RowsCols(dists, first, second) --> remove 2 closest
    local n = dists:size(1)
    dists = dists:cat( torch.Tensor(1, n):fill(self.inf), 1)
                                  :cat(torch.Tensor(n + 1, 1):fill(self.inf), 2)
    groups = agg:_merge2Groups(groups, first, second)
    for i = 1, dists:size(1) - 1 do --> update dists for only new cluster
      dists[i][dists:size(1)] = agg:_dist2Clusters(x, i, dists:size(1),
                                                        groups, distf, linkage)
    end
    first, second, md = agg:_minPoint(dists) --> max
  end
  return groups, dists, md
end

-------------------------------------------------------------------------------
--  END
-------------------------------------------------------------------------------

return M.agg
