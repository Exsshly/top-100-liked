# top-100-liked
LeetCode 热题 100 Java 常规题解

## 哈希

### 1.两数之和

```java
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            if (map.containsKey(target - nums[i]))
            	return new int[]{map.get(target - nums[i]), i};
            map.put(nums[i], i);
        }
        return new int[0];
    }
```

### 49.字母异位词分组

```java
    public List<List<String>> groupAnagrams(String[] strs) {
        Map<String, List<String>> map = new HashMap<>();
        for (String str : strs) {
            char[] chars = str.toCharArray();
            Arrays.sort(chars);
            String sortedStr = new String(chars);
            if (!map.containsKey(sortedStr))
            	map.put(sortedStr, new ArrayList<>());
            map.get(sortedStr).add(str);
        }
        return new ArrayList<>(map.values());
    }
```

### 128.最长连续序列

```java
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<>();
        for (int num : nums) set.add(num);
        int res = 0, cur = 0;
        for (int num : set) {
            if (set.contains(num - 1))
            	continue;
            cur = 1;
            int next = num + 1;
            while (set.contains(next)) {
                cur++;
                next++;
            }
            res = Math.max(res, cur);
        }
        return res;
    }
```

## 双指针

### 283.移动零

```java
    public void moveZeroes(int[] nums) {
        int count = 0;
        for (int num : nums)
            if (num == 0)
            	count++;
        int offset = 0;
        for (int num : nums)
            if (num != 0)
                nums[offset++] = num;
        while (count > 0)
            nums[nums.length - count--] = 0;
    }
```

### 11.盛最多水的容器

```java
    public int maxArea(int[] height) {
        int left = 0, right = height.length - 1;
        int res = 0;
        while (left < right) {
            int area = (right - left) * Math.min(height[left], height[right]);
            res = Math.max(res, area);
            if (height[left] < height[right]) left++;
            else right--;
        }
        return res;
    }
```

### 15.三数之和

```java
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        Arrays.sort(nums);
        if (nums[0] > 0 || nums[0] + nums[1] + nums[2] > 0)
        	return res;
        for (int i = 0; i < nums.length - 2; i++) {
            if (i > 0 && nums[i] == nums[i-1]) continue;
            int left = i + 1, right = nums.length - 1;
            while (left < right) {
                if (nums[i] + nums[left] + nums[right] == 0) {
                    List<Integer> inn = new ArrayList<>() {{
                    	add(nums[i]);
                    	add(nums[left]);
                    	add(nums[right]);
                    }};
                     /* List<Integer> inn
                     = new ArrayList<Integer>(Arrays.asList(nums[i], nums[left], nums[right]));
                     */
                    res.add(inn);
                    while (left < right && nums[left] == nums[left+1]) left++;
                    while (left < right && nums[right] == nums[right-1]) right--;
                    left++;
                    right--;
                } else if (nums[i] + nums[left] + nums[right] > 0) right--;
                else left++;
            }
        }
        return res;
    }
```

### 42.接雨水

```java
    public int trap(int[] height) {
        int res = 0, left = 0, right = height.length - 1;
        int leftHeightest = 0, rightHeightest = 0;
        while (left < right) {
            leftHeightest = Math.max(leftHeightest, height[left]);
            rightHeightest = Math.max(rightHeightest, height[right]);
            res += leftHeightest < rightHeightest ?
                leftHeightest - height[left++] : rightHeightest - height[right--];
        }
        return res;
    }
```

```java
    public int trap(int[] height) {
        int left = 0, right = height.length - 1;
        int leftHeightest = 0, rightHeightest = 0, res = 0;
        while (left < right) {
            leftHeightest = Math.max(leftHeightest, height[left]);
            rightHeightest = Math.max(rightHeightest, height[right]);
            if (leftHeightest < rightHeightest) {
                res += leftHeightest - height[left];
                left++;
            } else {
                res += rightHeightest - height[right];
                right--;
            }
        }
        return res;
    }
```

## 滑动窗口

### 3.无重复字符的最长子串

```java
    public int lengthOfLongestSubstring(String s) {
        Set<Character> set = new HashSet<>();
        int left = 0, right = 0, res = 0;
        while (right < s.length()) {
            if (!set.contains(s.charAt(right))) {
                set.add(s.charAt(right++));
                res = Math.max(res, right - left);
            } else set.remove(s.charAt(left++));
        }
        return res;
    }
```

### 438.找到字符串中所有字母异位词

```java
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> res = new ArrayList<>();
        int sLen = s.length(), pLen = p.length();
        if (sLen < pLen) return res;
        Map<Character, Integer> sMap = new HashMap<>();
        Map<Character, Integer> pMap = new HashMap<>();
        for (int i = 0; i < pLen; i++)
            pMap.put(p.charAt(i), pMap.getOrDefault(p.charAt(i), 0) + 1);
        for (int i = 0; i < pLen; i++)
            sMap.put(s.charAt(i), sMap.getOrDefault(s.charAt(i), 0) + 1);

        if (sMap.equals(pMap)) res.add(0);
        int left = 0, right = pLen - 1;
        while (right < sLen-1) {
            sMap.put(s.charAt(left), sMap.getOrDefault(s.charAt(left), 0) - 1);
            if (sMap.get(s.charAt(left)) == 0) sMap.remove(s.charAt(left));
            left++;
            right++;
            sMap.put(s.charAt(right), sMap.getOrDefault(s.charAt(right), 0) + 1);
            if (sMap.equals(pMap)) res.add(left);
        }
        return res;
    }
```

## 子串

### 560.和为K的子数组

```java
    public int subarraySum(int[] nums, int k) {
        int count = 0;
        int[] sums = new int[nums.length];
        Map<Integer, Integer> map = new HashMap<>();

        sums[0] = nums[0];
        for (int i = 1; i < nums.length; i++)
            sums[i] = nums[i] + sums[i-1];

        map.put(0, 1);

        for (int sum : sums) {
            count += map.getOrDefault(sum - k, 0);
            map.put(sum, map.getOrDefault(sum, 0) + 1);
        }
        
        return count;
    }
```

### 239.滑动窗口最大值

```java
    public int[] maxSlidingWindow(int[] nums, int k) {
        int[] res = new int[nums.length - k + 1];
        TreeMap<Integer, Integer> map = new TreeMap<>();
        for (int i = 0; i < k; i++)
        	map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        int idx = 0;
        res[idx++] = map.lastKey();
        int left = 0, right = k;
        while (right < nums.length) {
            map.put(nums[left], map.getOrDefault(nums[left], 0) - 1);
            if (map.get(nums[left]) == 0) map.remove(nums[left]);
            left++;
            map.put(nums[right], map.getOrDefault(nums[right], 0) + 1);
            right++;
            res[idx++] = map.lastKey();
        }
        return res;
    }
```

### 76.最小覆盖子串

```java
    public String minWindow(String s, String t) {
        if (s.length() < t.length()) return "";
        int[] tCount = new int[128];
        int[] wCount = new int[128];
        for (char ch : t.toCharArray()) tCount[ch]++;
        int required = 0;
        for (int count : tCount)
            if (count > 0) required++;
        int left = 0, right = 0;
        int formed = 0, minLen = Integer.MAX_VALUE, start = 0;
        while (right < s.length()) {
            char ch = s.charAt(right);
            wCount[ch]++;
            if (tCount[ch] > 0 && tCount[ch] == wCount[ch])
                formed++;
            while (left <= right && required == formed) {
                if (right - left + 1 < minLen) {
                    start = left;
                    minLen = right - left + 1;
                }
                ch = s.charAt(left);
                wCount[ch]--;
                if (tCount[ch] > 0 && wCount[ch] < tCount[ch])
                    formed--;
                left++;
            }
            right++;
        }
        return minLen == Integer.MAX_VALUE ? "" : s.substring(start,  start + minLen);
    }
```

## 普通数组

### 53.最大子数组和

```java
    public int maxSubArray(int[] nums) {
        int res = nums[0], sum = 0;
        for (int num : nums) {
            if (sum > 0) sum += num;
            else sum = num;
            res = Math.max(res, sum);
        }
        return res;
    }
```

### 56.合并区间

```java
    public int[][] merge(int[][] intervals) {
        Arrays.sort(intervals, (int[] a, int[] b) -> a[0] - b[0]);
        int cur[] = intervals[0];
        List<int[]> merged = new ArrayList<>();
        for (int i = 1; i < intervals.length; i++) {
            int[] next = intervals[i];
            if (next[0] <= cur[1]) {
                cur[1] = Math.max(cur[1], next[1]);
            } else {
                merged.add(cur);
                cur = next;
            }
        }
        merged.add(cur);
        int[][] res = new int[merged.size()][2];
        for (int i = 0; i < merged.size(); i++)
            res[i] = merged.get(i);
        return res;
    }
```

### 189.轮转数组

```java
    public void rotate(int[] nums, int k) {
        k %= nums.length;
        reverse(nums, 0, nums.length - 1);
        reverse(nums, 0, k - 1);
        reverse(nums, k, nums.length - 1);
    }
    public void reverse(int[] nums, int start, int end) {
        int T;
        while (start < end) {
            T = nums[start];
            nums[start] = nums[end];
            nums[end] = T;
            start++;
            end--;
        }
    }
```

### 238.除自身以外数组的乘积

```java
    public int[] productExceptSelf(int[] nums) {
        int[] L = new int[nums.length];
        int[] R = new int[nums.length];
        L[0] = 1;
        for (int i = 1; i < nums.length; i++)
            L[i] = L[i-1] * nums[i-1];
        R[nums.length-1] = 1;
        for (int i = nums.length - 2; i >= 0; i--)
            R[i] = R[i+1] * nums[i+1];
        int[] res = new int[nums.length];
        for (int i = 0; i < nums.length; i++)
            res[i] = L[i] * R[i];
        return res;
    }
```

### 41.缺失的第一个正数

```java
    public int firstMissingPositive(int[] nums) {
        int len = nums.length;
        for (int i = 0; i < len; i++) {
            while (nums[i] > 0 && nums[i] <= len && nums[nums[i] - 1] != nums[i]) {
                int temp = nums[i];
                nums[i] = nums[temp - 1];
                nums[temp - 1] = temp;
            }
        }
        for (int i = 0; i < len; i++) {
            if (nums[i] != i + 1) {
                return i + 1;
            }
        }
        return len + 1;
    }
```

## 矩阵

### 矩阵置零

```java
    public void setZeroes(int[][] matrix) {
        Set<Integer> rows = new HashSet<>();
        Set<Integer> cols = new HashSet<>();
        for (int i = 0; i < matrix.length; i++)
            for (int j = 0; j <matrix[0].length; j++)
                if (matrix[i][j] == 0) {
                    rows.add(i);
                    cols.add(j);
                }
        for (int i = 0; i < matrix.length; i++)
            for (int j = 0; j <matrix[0].length; j++)
                if (rows.contains(i) || cols.contains(j))
                    matrix[i][j] = 0;
    }
```

### 54.螺旋矩阵

```java
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<>();
        int top = 0, bottom = matrix.length - 1, left = 0, right = matrix[0].length - 1;
        while (true) {
            for (int i = left; i <= right; i++) res.add(matrix[top][i]);
            if (++top > bottom) break;
            for (int i = top; i <= bottom; i++) res.add(matrix[i][right]);
            if (--right < left) break;
            for (int i = right; i >= left; i--) res.add(matrix[bottom][i]);
            if (--bottom < top) break;
            for (int i = bottom; i >= top; i--) res.add(matrix[i][left]);
            if (++left > right) break;
        }
        return res;
    }
```

### 48.旋转图像

```java
    public void rotate(int[][] matrix) {
        int n = matrix.length;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][n - 1 - j];
                matrix[i][n - 1 - j] = temp;
            }
        }
    }
```

### 240.搜索二维矩阵Ⅱ

```java
    public boolean searchMatrix(int[][] matrix, int target) {
        int rows = matrix.length, cols = matrix[0].length;
        int row = 0, col = cols - 1;
        while (row < rows && col >= 0) {
            if (matrix[row][col] == target) return true;
            else if (matrix[row][col] > target) col--;
            else row++;
        }
        return false;
    }
```

## 链表

### 160.相交链表

```java
	public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
	    ListNode p = headA, q = headB;
	    while (p != q) {
	        p = p == null ? headB : p.next;
	        q = q == null ? headA : q.next;
	    }
	    return p;
	}
```

### 206.反转链表

```java
    public ListNode reverseList(ListNode head) {
        ListNode cur = head, pre = null;
        while(cur != null) {
            ListNode temp = cur.next;
            cur.next = pre;
            pre = cur;
            cur = temp;
        }
        return pre;
    }
```

### 234.回文链表

```java
    public boolean isPalindrome(ListNode head) {
        List<Integer> arr = new ArrayList<>();
        ListNode dummy = head;
        while (dummy != null) {
            arr.add(dummy.val);
            dummy = dummy.next;
        }
        for (int i = 0, j = arr.size() - 1; i < j; i++, j--)
            if (arr.get(i) != arr.get(j))
                return false;
        return true;
    }
```

### 141.环形链表

```java
    public boolean hasCycle(ListNode head) {
    	if (head == null || head.next == null) return false;
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            fast = fast.next.next;
            slow = slow.next;
            if (fast == slow)
            	return true;
        }
        return false;
    }
```

### 142.环形链表Ⅱ

```java
    public ListNode detectCycle(ListNode head) {
        if (head == null || head.next == null) return null;
        ListNode fast = head, slow = head;
        while (fast != null && fast.next != null) {
            slow = slow.next;
            fast = fast.next.next;
            if (fast == slow) {
                fast = head;
                while (fast != slow) {
                    fast = fast.next;
                    slow = slow.next;
                }
                return fast;
            }
        }
        return null;
    }
```

### 21.合并两个有序链表

```java
    public ListNode mergeTwoLists(ListNode list1, ListNode list2) {
        ListNode dummy = new ListNode(-1);
        ListNode head = dummy;
        while (list1 != null && list2 != null) {
            if (list1.val <= list2.val) {
                dummy.next = list1;
                list1 = list1.next;
            } else {
                dummy.next = list2;
                list2 = list2.next;
            }
            dummy = dummy.next;
        }
        dummy.next = list1 == null ? list2 : list1;
        return head.next;
    }
```

### 2.两数相加

```java
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode dummy = new ListNode(-1), head = dummy;
        int carry = 0;
        while (l1 != null || l2 != null || carry != 0) {
            int sum = carry;
            if (l1 != null) {
                sum += l1.val;
                l1 = l1.next;
            }
            if (l2 != null) {
                sum += l2.val;
                l2 = l2.next;
            }
            carry = sum / 10;
            sum %= 10;
            dummy.next = new ListNode(sum);
            dummy = dummy.next;
        }
        return head.next;
    }
```

### 19.删除链表的倒数第N个结点

```java
    public ListNode removeNthFromEnd(ListNode head, int n) {
        ListNode dummy = new ListNode(-1);
        dummy.next = head;
        ListNode fast = dummy, slow = dummy;
        while (n-- > 0)
            fast = fast.next;
        while (fast.next != null) {
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return dummy.next;
    }
```

### 24.两两交换链表中的节点

```java
    public ListNode swapPairs(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode node0 = dummy;
        while (node0.next != null && node0.next.next != null) {
            ListNode node1 = node0.next;
            ListNode node2 = node0.next.next;
            node0.next = node2;
            node1.next = node2.next;
            node2.next = node1;
            node0 = node1;
        }
        return dummy.next;
    }
```

### 25.K个一组翻转链表

```java
	// 非递归
    public ListNode reverseKGroup(ListNode head, int k) {
        ListNode vhead = new ListNode(-1);
        vhead.next = head;

        ListNode pre = vhead;
        ListNode end = vhead;
        while (end.next != null) {
            for (int i = 0; i < k && end != null; i++) end = end.next;
            if (end == null) break;
            ListNode start = pre.next;
            ListNode next = end.next;
            end.next = null;
            pre.next = reverse(start);
            start.next = next;
            pre = start;
            end = pre;
        }
        return vhead.next;
    }
    public ListNode reverse(ListNode head) {
        ListNode pre = null;
        ListNode cur = head;
        while(cur != null) {
            ListNode next = cur.next;
            cur.next = pre;
            pre = cur;
            cur = next;
        }
        return pre;
    }
```

```java
	// 递归
	public ListNode reverseKGroup(ListNode head, int k) {
        if (head == null) return null;
        ListNode tail = head;
        for (int i = 0; i < k; i++) {
            if (tail == null) return head;
            tail = tail.next;
        }
        ListNode newHead = reverse(head, tail);
        head.next = reverseKGroup(tail, k);
        return newHead;
    }

    public ListNode reverse(ListNode head, ListNode tail) {
        ListNode pre = null;
        while (head != tail) {
            ListNode next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
```

### 138.随机链表的复制

```java
    public Node copyRandomList(Node head) {
        if (head == null) return null;
        Map<Node, Node> map = new HashMap<>();
        Node cur = head;
        while (cur != null) {
            map.put(cur, new Node(cur.val));
            cur = cur.next;
        }
        cur = head;
        while (cur != null) {
            Node node = map.get(cur);
            node.next = map.get(cur.next);
            node.random = map.get(cur.random);
            cur = cur.next;
        }
        return map.get(head);
    }
```

### 148.排序链表

```java
	// logN 空间
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        ListNode fast = head, slow = head;
        while (fast.next != null && fast.next.next != null) {
            fast = fast.next.next;
            slow = slow.next;
        }
        ListNode idx = slow.next;
        slow.next = null;
        ListNode left = sortList(head);
        ListNode right = sortList(idx);
        ListNode dummy = new ListNode(-1000000);
        ListNode p = dummy;
        while (left != null && right != null) {
            if (left.val <= right.val){
                p.next = left;
                left = left.next;
            } else {
                p.next = right;
                right = right.next;
            }
            p = p.next;
        }
        p.next = left == null ? right : left;
        return dummy.next;
    }
```

```java
	// 常数空间
    public ListNode sortList(ListNode head) {
        if (head == null || head.next == null) return head;
        int len = 0;
        ListNode curr = head;
        while (curr != null) {
            len++;
            curr = curr.next;
        }
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        for (int step = 1; step < len; step <<= 1) {
            ListNode tail = dummy;
            ListNode curr = dummy.next;
            while (curr != null) {
                ListNode left = curr;
                ListNode right = split(left, step);
                curr = split(right, step);
                tail = merge(left, right, tail);
            }
        }
        return dummy.next;
    }
    private ListNode split(ListNode head, int k) {
        for (int i = 1; head != null && i < k; i++) {
            head = head.next;
        }
        if (head == null) return null;
        ListNode next = head.next;
        head.next = null;
        return next;
    }
    private ListNode merge(ListNode l1, ListNode l2, ListNode tail) {
        ListNode curr = tail;
        while (l1 != null && l2 != null) {
            if (l1.val <= l2.val) {
                curr.next = l1;
                l1 = l1.next;
            } else {
                curr.next = l2;
                l2 = l2.next;
            }
            curr = curr.next;
        }
        curr.next = (l1 != null) ? l1 : l2;
        while (curr.next != null) curr = curr.next;
        return curr;
    }
```

### 23.合并K个升序链表

```java
    public ListNode mergeKLists(ListNode[] lists) {
        PriorityQueue<ListNode> pq = new PriorityQueue<>((a, b) -> a.val - b.val);
        ListNode dummy = new ListNode(-1), cur = dummy;
        for (ListNode node : lists)
            if (node != null)
                pq.offer(node);
        while (!pq.isEmpty()) {
            cur.next = pq.poll();
            cur = cur.next;
            if (cur.next != null)
                pq.offer(cur.next);
        }
        return dummy.next;
    }
```

### 146.LRU缓存

```java
    class LRUCache {
        private final int capacity;
        private final Map<Integer, Integer> cache = new LinkedHashMap<>();

        public LRUCache(int capacity) {
            this.capacity = capacity;
        }

        public int get(int key) {
            Integer value = cache.remove(key);
            if (value != null) {
                cache.put(key, value);
                return value;
            }
            return -1;
        }

        public void put(int key, int value) {
            if (cache.remove(key) != null) {
                cache.put(key, value);
                return;
            }
            if (cache.size() == capacity) {
                Integer eldestKey = cache.keySet().iterator().next();
                cache.remove(eldestKey);
            }
            cache.put(key, value);
        }
    }
```

## 二叉树

### 94.二叉树的中序遍历

```java
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        inorder(root, res);
        return res;
    }
    public void inorder(TreeNode root, List<Integer> res) {
        if (root == null) return;
        inorder(root.left, res);
        res.add(root.val);
        inorder(root.right, res);
    }
```

### 104.二叉树的最大深度

```java
    int res = 0;
    public int maxDepth(TreeNode root) {
        order(root, 0);
        return res;
    }
    public void order(TreeNode root, int depth) {
        if (root == null) {
            res = Math.max(res, depth);
            return;
        }
        order(root.left, depth + 1);
        order(root.right, depth + 1);
    }
```

### 226.翻转二叉树

```java
    public TreeNode invertTree(TreeNode root) {
        if (root == null) return null;

        TreeNode node = root.left;
        root.left = root.right;
        root.right = node;
        invertTree(root.left);
        invertTree(root.right);
        return root;
    }
```

### 101.对称二叉树

```java
    public boolean isSymmetric(TreeNode root) {
        return check(root.left, root.right);
    }
    public boolean check(TreeNode p, TreeNode q) {
        if (p == null && q == null) return true;
        if (p == null || q == null) return false;
        return p.val == q.val && check(p.left, q.right) && check(p.right, q.left);
    }
```

### 543.二叉树的直径

```java
    int res = 0;
    public int diameterOfBinaryTree(TreeNode root) {
        order(root);
        return res;
    }

    public int order(TreeNode root) {
        if (root == null) return 0;
        int leftLen = order(root.left);
        int rightLen = order(root.right);
        res = Math.max(res, leftLen + rightLen);
        return Math.max(leftLen, rightLen) + 1;
    }
```

### 102.二叉树的层序遍历

```java
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> res = new ArrayList<>();
        Deque<TreeNode> que = new ArrayDeque<>();
        if (root == null) return res;
        que.addLast(root);
        while (!que.isEmpty()) {
            int size = que.size();
            List<Integer> inner = new ArrayList<>();
            while (size-- > 0) {
                TreeNode node = que.pollFirst();
                inner.add(node.val);
                if (node.left != null) que.add(node.left);
                if (node.right != null) que.add(node.right);
            }
            res.add(inner);
        }
        return res;
    }
```

### 108.将有序数组转换为二叉搜索树

```java
    public TreeNode sortedArrayToBST(int[] nums) {
        return buildBST(nums, 0, nums.length - 1);
    }

    public TreeNode buildBST(int[] nums, int left, int right) {
        if (left > right) return null;
        int mid = left + (right - left) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = buildBST(nums, left, mid - 1);
        root.right = buildBST(nums, mid + 1, right);
        return root;
    }
```

### 98.验证二叉搜索树

```java
    TreeNode pre = null;
    public boolean isValidBST(TreeNode root) {
        return inorder(root);
    }
    public boolean inorder(TreeNode root) {
        if (root == null) return true;
        if (!inorder(root.left)) return false;
        if (pre != null && pre.val >= root.val) return false;
        pre = root;
        return inorder(root.right);
    }
```

### 230.二叉搜索树中第K小的元素

```java
    int res, count;
    public int kthSmallest(TreeNode root, int k) {
        count = k;
        inorder(root);
        return res;
    }
    public void inorder(TreeNode root) {
        if (root == null) return;
        inorder(root.left);
        if (--count == 0) {
            res = root.val;
            return;
        }
        inorder(root.right);
    }
```

### 199.二叉树的右视图

```java
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new ArrayList<>();
        Deque<TreeNode> deque = new ArrayDeque<>();
        if (root == null) return res;
        deque.addLast(root);
        while (!deque.isEmpty()) {
            int size = deque.size();
            TreeNode node = new TreeNode();
            while (size-- > 0) {
                node = deque.pollFirst();
                if (node.left != null) deque.addLast(node.left);
                if (node.right != null) deque.addLast(node.right);
            }
            res.add(node.val);
        }
        return res;
    }
```

### 114.二叉树展开为链表

```java
	// O(1)
    public void flatten(TreeNode root) {
        TreeNode cur = root;
        while (cur != null) {
            if (cur.left != null) {
                TreeNode next = cur.left;
                TreeNode pre = next;
                while (pre.right != null) pre = pre.right;
                pre.right = cur.right;
                cur.left = null;
                cur.right = next;
            }
            cur = cur.right;
        }
    }
```
    
```java
	// O(n)
	List<Integer> arr = new ArrayList<>();
    public void flatten(TreeNode root) {
        preorder(root);
        TreeNode cur = root;
        for (int i = 1; i < arr.size(); i++) {
            cur.left = null;
            cur.right = new TreeNode(arr.get(i));
            cur = cur.right;
        }
    }
    public void preorder(TreeNode root) {
        if (root == null) return;
        arr.add(root.val);
        preorder(root.left);
        preorder(root.right);
    }
```

### 105.从前序与中序遍历序列构造二叉树

```java
    public TreeNode buildTree(int[] preorder, int[] inorder) {
        if (preorder == null || inorder == null || preorder.length == 0 || inorder.length == 0) return null;
        TreeNode root = new TreeNode(preorder[0]);
        int inorderRootIndex = 0;
        for (int i = 0; i < inorder.length; i++) {
            if (inorder[i] == preorder[0]) {
                inorderRootIndex = i;
                break;
            }
        }
        root.left = buildTree(subArray(preorder, 1, inorderRootIndex + 1),
        	subArray(inorder, 0, inorderRootIndex));
        root.right = buildTree(subArray(preorder, inorderRootIndex + 1, preorder.length),
        	subArray(inorder, inorderRootIndex + 1, inorder.length));
        return root;
    }
    public int[] subArray(int[] array, int start, int end) {
        int[] arr = new int[end - start];
        System.arraycopy(array, start, arr, 0, end - start);
        return arr;
    }
```

### 112.路径总和Ⅰ

```java
    public boolean hasPathSum(TreeNode root, int targetSum) {
        if (root == null) return false;
        targetSum -= root.val;
        if (root.left == null && root.right == null)
        	return targetSum == 0;
        return hasPathSum(root.left, targetSum) || hasPathSum(root.right, targetSum);
    }
```

### 113.路径总和Ⅱ

```java
    List<Integer> path = new ArrayList<>();
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> pathSum(TreeNode root, int targetSum) {
        recur(root, targetSum);
        return res;
    }
    public void recur(TreeNode root, int targetSum) {
        if (root == null) return;
        path.add(root.val);
        targetSum -= root.val;
        if (targetSum == 0 && root.left == null && root.right == null)
            res.add(new ArrayList<Integer>(path));
        recur(root.left, targetSum);
        recur(root.right, targetSum);
        path.removeLast();
    }
```

### 437.路径总和Ⅲ

```java
    public int pathSum(TreeNode root, int targetSum) {
        if (root == null) return 0;
        int res = countPath(root, targetSum);
        res += pathSum(root.left, targetSum);
        res += pathSum(root.right, targetSum);
        return res;
    }
    public int countPath(TreeNode node, int targetSum) {
        if (node == null) return 0;
        int count = 0;
        if (targetSum == node.val)
            count++;
        count += countPath(node.left, targetSum - node.val);
        count += countPath(node.right, targetSum - node.val);
        return count;
    }
```

### 236.二叉树的最近公共祖先

```java
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || root == p || root == q) return root;
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        if (left != null && right != null)
            return root;
        return left == null ? right : left;
    }
```

### 124.二叉树中的最大路径和

```java
    int res = Integer.MIN_VALUE;
    public int maxPathSum(TreeNode root) {
        helper(root);
        return res;
    }
    public int helper(TreeNode root) {
        if (root == null) return 0;
        int leftMax = Math.max(helper(root.left), 0);
        int rightMax = Math.max(helper(root.right), 0);
        res = Math.max(res, root.val + leftMax + rightMax);
        return root.val + Math.max(leftMax, rightMax);
    }
```

## 图论

### 200.岛屿数量

```java
    public int numIslands(char[][] grid) {
        int count = 0;
        boolean[][] isVisited = new boolean[grid.length][grid[0].length];
        for (int i = 0; i < grid.length; i++) {
            for (int j = 0; j < grid[0].length; j++) {
                if (grid[i][j] == '1') {
                    count++;
                    dfs(grid, i, j);
                }
            }
        }
        return count;
    }
    public void dfs(char[][] grid, int i, int j) {
        grid[i][j] = '0';
        if (i - 1 >= 0 && grid[i-1][j] == '1')
            dfs(grid, i - 1, j);
        if (i + 1 < grid.length && grid[i+1][j] == '1')
            dfs(grid, i + 1, j);
        if (j - 1 >= 0 && grid[i][j-1] == '1')
            dfs(grid, i, j - 1);
        if (j + 1 < grid[0].length && grid[i][j+1] == '1')
            dfs(grid, i, j + 1);
    }
```

### 994.腐烂的橘子

```java
    public int orangesRotting(int[][] grid) {
        Deque<int[]> que = new ArrayDeque<>();
        int m = grid.length, n = grid[0].length;
        int time = 0, good = 0;
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                if (grid[i][j] == 2)
                    que.addLast(new int[]{i, j});
                else if (grid[i][j] == 1)
                    good++;
            }
        }
        if (que.isEmpty()) return good == 0 ? 0 : -1;
        while (!que.isEmpty()) {
            boolean flag = false;
            int size = que.size();
            while (size-- > 0) {
                int[] rotPos = que.pollFirst();
                int i = rotPos[0], j = rotPos[1];
                if (i - 1 >= 0 && grid[i-1][j] == 1) {
                    grid[i-1][j] = 2;
                    que.addLast(new int[]{i-1, j});
                    flag = true;
                }
                if (i + 1 < m && grid[i+1][j] == 1) {
                    grid[i+1][j] = 2;
                    que.addLast(new int[]{i+1, j});
                    flag = true;
                }
                if (j - 1 >= 0 && grid[i][j-1] == 1) {
                    grid[i][j-1] = 2;
                    que.addLast(new int[]{i, j - 1});
                    flag = true;
                }
                if (j + 1 < n && grid[i][j+1] == 1) {
                    grid[i][j+1] = 2;
                    que.addLast(new int[]{i, j + 1});
                    flag = true;
                }
            }
            if (flag == true) time++;
        }
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                if (grid[i][j] == 1)
                    return -1;
        if (time == 0 && good == 0)
            return 0;
        return time;
    }
```

### 207.课程表

```java
    public boolean canFinish(int numCourses, int[][] prerequisites) {
        int[] inDegree = new int[numCourses];
        List<List<Integer>> postCourse = new ArrayList<>();
        for (int i = 0; i < numCourses; i++)
            postCourse.add(new ArrayList<>());
        for (int[] prerequisite : prerequisites) {
            int course = prerequisite[0];
            int prerequisiteCourse = prerequisite[1];
            inDegree[course]++;
            postCourse.get(prerequisiteCourse).add(course);
        }
        Deque<Integer> que = new ArrayDeque<>();
        for (int i = 0; i < numCourses; i++)
            if (inDegree[i] == 0)
                que.offer(i);
        int completedCoursesCount = 0;
        while (!que.isEmpty()) {
            int currentCourse = que.poll();
            completedCoursesCount++;
            List<Integer> nextCourses = postCourse.get(currentCourse);
            for (int nextCourse : nextCourses) {
                inDegree[nextCourse]--;
                if (inDegree[nextCourse] == 0) {
                    que.offer(nextCourse);
                }
            }
        }
        return completedCoursesCount == numCourses;
    }
```

### 208.实现前缀树

```java
    class TireNode {
        private boolean isEnd;
        TireNode[] next;

        public TireNode() {
            isEnd = false;
            next = new TireNode[26];
        }
    }

    private TireNode root;

    public Trie() {
        root = new TireNode();
    }

    public void insert(String word) {
        TireNode node = root;
        for (char c : word.toCharArray()) {
            if (node.next[c - 'a'] == null) {
                node.next[c - 'a'] = new TireNode();
            }
            node = node.next[c - 'a'];
        }
        node.isEnd = true;
    }

    public boolean search(String word) {
        TireNode node = root;
        for (char c : word.toCharArray()) {
            node = node.next[c - 'a'];
            if (node == null) {
                return false;
            }
        }
        return node.isEnd;
    }

    public boolean startsWith(String prefix) {
        TireNode node = root;
        for (char c : prefix.toCharArray()) {
            node = node.next[c - 'a'];
            if (node == null) {
                return false;
            }
        }
        return true;
    }
```

## 回溯

### 46.全排列

```java
    List<Integer> path = new ArrayList<>();
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> permute(int[] nums) {
        boolean[] isVisited = new boolean[nums.length];
        backtracking(nums, isVisited);
        return res;
    }
    public void backtracking(int[] nums, boolean[] isVisited) {
        if (path.size() == nums.length) res.add(new ArrayList<>(path));
        for (int i = 0; i < nums.length; i++) {
            if (isVisited[i] == true) continue;
            isVisited[i] = true;
            path.add(nums[i]);
            backtracking(nums, isVisited);
            path.removeLast();
            isVisited[i] = false;
        }
    }
```

### 78.子集

```java
    List<Integer> path = new ArrayList<>();
    List<List<Integer>> res = new ArrayList<>();
    public List<List<Integer>> subsets(int[] nums) {
        backtracking(nums, 0);
        return res;
    }
    public void backtracking(int[] nums, int startIndex) {
        res.add(new ArrayList<>(path));
        if (startIndex >= nums.length) return;
        for (int i = startIndex; i < nums.length; i++) {
            path.add(nums[i]);
            backtracking(nums, i+1);
            path.removeLast();
        }
    }
```

### 17.电话号码的字母组合

```java
    List<String> res = new ArrayList<>();
    String[] keyboard = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    StringBuilder sb = new StringBuilder();
    public List<String> letterCombinations(String digits) {
        if (digits.length() == 0) return res;
        backtracking(digits, 0);
        return res;
    }
    public void backtracking(String digits, int pos) {
        if (pos == digits.length()) {
            res.add(sb.toString());
            return;
        }
        String key = keyboard[digits.charAt(pos) - '0'];
        for (int i = 0; i < key.length(); i++) {
            sb.append(key.charAt(i));
            backtracking(digits, pos + 1);
            sb.deleteCharAt(sb.length() - 1);
        }
    }
```

### 39.组合总和

```java
	List<Integer> path = new ArrayList<>();
    List<List<Integer>> res = new ArrayList<>();    
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        backtrack(candidates, target, 0, 0);
        return res;
    }

    public void backtrack(int[] candidates, int target, int curSum, int idx) {
        if (curSum == target)
            res.add(new ArrayList<>(path));
        for (int i = idx; i < candidates.length; i++) {
            if (curSum+candidates[i] > target) continue;
            path.add(candidates[i]);
            backtrack(candidates, target, curSum+candidates[i], i);
            path.removeLast();
        }
    }
```

### 22.括号生成

```java
    List<String> res = new ArrayList<>();
    int left, right;
    StringBuilder sb = new StringBuilder();
    public List<String> generateParenthesis(int n) {
        backtrack(n);
        return res;
    }
    public void backtrack(int n) {
        if (right == n) res.add(sb.toString());
        if (left < n) {
            left++;
            sb.append("(");
            backtrack(n);
            sb.deleteCharAt(sb.length()-1);
            left--;
        }
        if (right < left) {
            right++;
            sb.append(")");
            backtrack(n);
            sb.deleteCharAt(sb.length()-1);
            right--;
        }
    }
```

### 79.单词搜索

```java
    public boolean exist(char[][] board, String word) {
        boolean[][] isVisited = new boolean[board.length][board[0].length];
        int len = word.length();
        for (int i = 0; i < board.length; i++) {
            for (int j = 0; j < board[0].length; j++) {
                if (dfs(board, isVisited, word, len, i, j, 0))
                    return true;
            }
        }
        return false;
    }
    public boolean dfs(char[][] board, boolean[][] isVisited, String word, int len, int i, int j, int idx) {
        if (idx == len) return true;
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length ||
                isVisited[i][j] == true || board[i][j] != word.charAt(idx)) return false;
        isVisited[i][j] = true;
        if (dfs(board, isVisited, word, len, i - 1, j, idx+1)) return true;
        if (dfs(board, isVisited, word, len, i + 1, j, idx+1)) return true;
        if (dfs(board, isVisited, word, len, i, j - 1, idx+1)) return true;
        if (dfs(board, isVisited, word, len, i, j + 1, idx+1)) return true;
        isVisited[i][j] = false;
        return false;
    }
```

### 131.分割回文串

```java
    List<List<String>> res = new ArrayList<>();
    Deque<String> que = new ArrayDeque<>();
    int length = 0;
    public List<List<String>> partition(String s) {
        length = s.length();
        backtracking(s, 0);
        return res;
    }
    public void backtracking(String s, int idx) {
        if (idx >= length) res.add(new ArrayList<>(que));
        for (int i = idx; i < length; i++) {
            if (judge(s, idx, i)) {
                String str = s.substring(idx, i+1);
                que.addLast(str);
            } else continue;
            backtracking(s, i + 1);
            que.removeLast();
        }
    }
    public boolean judge(String s, int p, int q) {
        while (p < q)
            if (s.charAt(p++) != s.charAt(q--))
                return false;
        return true;
    }
```

### 51.N 皇后

```java
    List<List<String>> res = new ArrayList<>();
    public List<List<String>> solveNQueens(int n) {
        char[][] board = new char[n][n];
        for (char[] ch : board) Arrays.fill(ch, '.');
        res = new ArrayList<>();
        backtrack(n, 0, board);
        return res;
    }
    public void backtrack(int n, int row, char[][] board) {
        if (row == n) {
            res.add(Array2List(board));
            return;
        }
        for (int col = 0; col < n; ++col) {
            if (valid(row, col, n, board)) {
                board[row][col] = 'Q';
                backtrack(n, row+1, board);
                board[row][col] = '.';
            }
        }
    }
    public List<String> Array2List(char[][] board) {
        List<String> strList = new ArrayList<>();
        for (char[] row : board) {
            strList.add(new String(row));
        }
        return strList;
    }
    public boolean valid(int row, int col, int n, char[][] board) {
        for (int i = 0; i < row; i++)
            if (board[i][col] == 'Q')
                return false;
        for (int i = row - 1, j = col - 1; i >= 0 && j >=0; i--, j--)
            if (board[i][j] == 'Q')
                return false;
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++)
            if (board[i][j] == 'Q')
                return false;
        return true;
    }
```

## 二分查找

### 35.搜索插入位置

```java
    public int searchInsert(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] > target) right = mid - 1;
            else left = mid + 1;
        }
        return left;
    }
```

### 74.搜索二维矩阵

```java
    public boolean searchMatrix(int[][] matrix, int target) {
        int rows = matrix.length, cols = matrix[0].length;
        int row = 0, col = cols - 1;
        // for (int row = 0, col = cols - 1; row < rows && col >=0; ) {
        while (row < rows && col >= 0) {
            if (matrix[row][col] == target) return true;
            else if (matrix[row][col] > target) col--;
            else row++;
        }
        return false;
    }
```

### 34.在排序数组中查找元素的第一个和最后一个位置

```java
    public int[] searchRange(int[] nums, int target) {
        int left = 0, right = nums.length - 1;
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) {
                int i = mid;
                while (i >= 0 && nums[i] == target) i--;
                int j = mid;
                while (j < nums.length && nums[j] == target) j++;
                return new int[]{++i, --j};
            } else if (nums[mid] > target) right = mid - 1;
            else left = mid + 1;
        }
        return new int[]{-1, -1};
    }
```

### 33.搜索旋转排序数组

```java
	public int search(int[] nums, int target) {
        int idx = findMin(nums);
        if (target > nums[nums.length - 1]) return find(nums, 0, idx - 1, target);
        return find(nums, idx, nums.length - 1, target);
    }
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[right])
            	right = mid;
            else
            	left = mid + 1;
        }
        return left;
    }
    public int find(int[] nums, int left, int right, int target) {
        while (left <= right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] == target) return mid;
            else if (nums[mid] > target) right = mid - 1;
            else left = mid + 1;
        }
        return -1;
    }
```

### 153.寻找旋转排序数组中的最小值

```java
    public int findMin(int[] nums) {
        int left = 0, right = nums.length - 1;
        while (left < right) {
            int mid = left + (right - left) / 2;
            if (nums[mid] < nums[right])
                right = mid;
            else
                left = mid + 1;
        }
        return nums[left];
    }
```

### 4.寻找两个正序数组的中位数

```java
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int m = nums1.length, n = nums2.length;
        int[] merged = new int[m + n];
        for (int i = 0; i < m; i++) merged[i] = nums1[i];
        for (int i = 0; i < n; i++) merged[m + i] = nums2[i];
        Arrays.sort(merged);
        if ((m + n) % 2 == 0)
        	return (merged[(m + n) / 2 - 1] + merged[(m + n) / 2]) / 2.0;
        else
        	return merged[(m + n) / 2];
    }
```

```java
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int len = len1 + len2;

        if (len % 2 != 0) return findK(nums1, nums2, len / 2 + 1);
        else return (findK(nums1, nums2, len / 2) + findK(nums1, nums2, len / 2 + 1)) / 2.0;
    }
    private int findK(int[] nums1, int[] nums2, int k) {
        int len1 = nums1.length;
        int len2 = nums2.length;
        int index1 = 0;
        int index2 = 0;

        while (true) {
            if (index1 == len1) return nums2[index2 + k - 1];
            if (index2 == len2) return nums1[index1 + k - 1];

            if (k == 1) return Math.min(nums1[index1], nums2[index2]);

            int newIndex1 = Math.min(index1 + k / 2, len1) - 1;
            int newIndex2 = Math.min(index2 + k / 2, len2) - 1;

            if (nums1[newIndex1] <= nums2[newIndex2]) {
                k -= (newIndex1 - index1 + 1);
                index1 = newIndex1 + 1;
            } else {
                k -= (newIndex2 - index2 + 1);
                index2 = newIndex2 + 1;
            }
        }
    }
```

## 栈

### 20.有效的括号

```java
    public boolean isValid(String s) {
        Deque<Character> stack = new ArrayDeque<>();
        int i = 0;
        while (i < s.length()) {
            char ch = s.charAt(i++);
            if (ch == '(') stack.push(')');
            else if (ch == '{') stack.push('}');
            else if (ch == '[') stack.push(']');
            else if (stack.isEmpty() || ch != stack.pop()) return false;
        }
        return true;
    }
```

### 155.最小栈

```java
    private Stack<Integer> stack;
    private Stack<Integer> minStack;

    public MinStack() {
        stack = new Stack<>();
        minStack = new Stack<>();
    }

    public void push(int val) {
        stack.push(val);
        if (minStack.isEmpty() || val <= minStack.peek())
            minStack.push(val);
    }

    public void pop() {
        if (stack.pop() == minStack.peek())
            minStack.pop();
    }

    public int top() {
        return stack.peek();
    }

    public int getMin() {
        return minStack.peek();
    }
```

### 394.字符串解码

```java
    public String decodeString(String s) {
        Stack<Integer> count = new Stack<>();
        Stack<String> str = new Stack<>();
        int num = 0;
        String cur = "";
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            if (ch >= '0' && ch <= '9')
            	num = num * 10 + ch - '0';
            else if (ch == '[') {
                count.push(num);
                num = 0;
                str.push(cur);
                cur = "";
            } else if (ch == ']') {
                int times = count.pop();
                StringBuilder sb = new StringBuilder(str.pop());
                while (times-- > 0) {
                	sb.append(cur);
                }
                cur = sb.toString();
            } else {
            	cur += ch;
            	}
        }
        return cur;
    }
```

### 739.每日温度

```java
    public int[] dailyTemperatures(int[] temperatures) {
        Stack<Integer> stack = new Stack<>();
        int[] res = new int[temperatures.length];
        for (int i = 0; i < temperatures.length; i++) {
            while (!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int idx = stack.pop();
                res[idx] = i - idx;
            }
            stack.push(i);
        }
        return res;
    }
```

### 84.柱状图中最大的矩形

```java
    public int largestRectangleArea(int[] heights) {
        int res = 0;
        int[] newHeights = new int[heights.length + 2];
        for (int i = 0; i < heights.length; i++) newHeights[i+1] = heights[i];
        Stack<Integer> stack = new Stack<>();
        for (int i = 0; i < newHeights.length; i++) {
            while (!stack.isEmpty() && newHeights[i] < newHeights[stack.peek()]) {
                int idx = stack.pop(), left = stack.peek(), right = i;
                res = Math.max(res, newHeights[idx] * (right - left - 1));
            }
            stack.push(i);
        }
        return res;
    }
```

## 堆

### 215.数组中的第K个最大元素

```java
    public int findKthLargest(int[] nums, int k) {
        PriorityQueue<Integer> pq = new PriorityQueue<>((a, b) -> b - a);
        for (int num : nums)
            pq.add(num);
        while (k-- > 1) pq.poll();
        return pq.poll();
    }
```

### 347.前K个高频元素

```java
    public int[] topKFrequent(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<>();
        for (int num : nums)
            map.put(num, map.getOrDefault(num, 0) + 1);
        PriorityQueue<int[]> pq = new PriorityQueue<>((a, b) -> b[1] = a[1]);
        for (Map.Entry<Integer, Integer> entry : map.entrySet())
            pq.add(new int[]{entry.getKey(), entry.getValue()});
        int[] res = new int[k];
        while (k-- > 0)
            res[k] = pq.poll()[0];
        return res;
    }
```

### 295.数据流的中位数

```java
    PriorityQueue<Integer> maxPQ;
    PriorityQueue<Integer> minPQ;
    public MedianFinder() {
        maxPQ = new PriorityQueue<>((a, b) -> b - a);
        minPQ = new PriorityQueue<>();
    }
    public void addNum(int num) {
        if (maxPQ.isEmpty() || num < maxPQ.peek())
            maxPQ.add(num);
        else
            minPQ.add(num);

        if (maxPQ.size() > minPQ.size() + 1)
            minPQ.add(maxPQ.poll());
        else if (minPQ.size() > maxPQ.size() + 1)
            maxPQ.add(minPQ.poll());
    }
    public double findMedian() {
        if (maxPQ.size() == minPQ.size())
            return (maxPQ.peek() + minPQ.peek()) / 2.0;
        else if (maxPQ.size() > minPQ.size())
            return maxPQ.peek();
        else
            return minPQ.peek();
    }
```

## 贪心算法

### 121.买卖股票的最佳时机

```java
    public int maxProfit(int[] prices) {
        int cost = Integer.MAX_VALUE, profit = 0;
        for (int price : prices) {
            cost = Math.min(cost, price);
            profit = Math.max(profit, price - cost);
        }
        return profit;
    }
```

### 55.跳跃游戏

```java
    public boolean canJump(int[] nums) {
        int cover = 0;
        for (int i = 0; i <= cover; i++) {
            cover = Math.max(cover, i + nums[i]);
            if (cover >= nums.length - 1) return true;
        }
        return false;
    }
```

### 45.跳跃游戏Ⅱ

```java
    public int jump(int[] nums) {
        int count = 0, cover = 0, pos = 0;
        for (int i = 0; i < nums.length - 1; i++) {
            cover = Math.max(cover, i + nums[i]);
            if (i == pos) {
                pos = cover;
                count++;
            }
        }
        return count;
    }
```

### 763.划分字母区间

```java
    public List<Integer> partitionLabels(String s) {
        List<Integer> res = new ArrayList<>();
        int[] lastPos = new int[26];
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            lastPos[ch - 'a'] = i;
        }
        int start = 0, end = 0;
        for (int i = 0; i < s.length(); i++) {
            char ch = s.charAt(i);
            end = Math.max(end, lastPos[ch - 'a']);
            if (i == end) {
                res.add(end - start + 1);
                start = end + 1;
            }
        }
        return res;
    }
```

## 动态规划

### 70.爬楼梯

```java
    public int climbStairs(int n) {
        if (n == 1) return 1;
        int[] dp = new int[n + 1];
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i <= n; i++)
            dp[i] = dp[i-1] + dp[i-2];
        return dp[n];
    }
```

### 118.杨辉三角

```java
	public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<>();
        if (numRows == 0) return res;

        List<Integer> row1 = new ArrayList<>();
        row1.add(1);
        res.add(row1);
        if (numRows == 1) return res;

        List<Integer> row2 = new ArrayList<>();
        row2.add(1);
        row2.add(1);
        res.add(row2);

        for (int i = 2; i < numRows; i++) {
            List<Integer> preRow = res.get(i-1);
            List<Integer> curRow = new ArrayList<>();
            curRow.add(1);
            for (int j = 1; j < preRow.size(); j++) {
                int sum = preRow.get(j - 1) + preRow.get(j);
                curRow.add(sum);
            }
            curRow.add(1);
            res.add(curRow);
        }
        return res;
    }
```

### 198.打家劫舍

```java
    public int rob(int[] nums) {
        if (nums.length == 0) return 0;
        if (nums.length == 1) return nums[0];
        int[] dp = new int[nums.length];

        dp[0] = nums[0];
        dp[1] = Math.max(nums[0], nums[1]);
        for (int i = 2; i < nums.length; i++)
            dp[i] = Math.max(dp[i-1], dp[i-2] + nums[i]);
        return dp[nums.length-1];
    }
```

### 279.完全平方数

```java
    public int numSquares(int n) {
        int[] dp = new int[n + 1];
        Arrays.fill(dp, Integer.MAX_VALUE);
        dp[0] = 0;
        for (int i = 1; i <= n; i++)
            for (int j = 1; j*j <= i; j++)
                dp[i] = Math.min(dp[i], dp[i-j*j] + 1);
        return dp[n];
    }
```

### 322.零钱兑换

```java
    public int coinChange(int[] coins, int amount) {
        int[] dp = new int[amount + 1];
        Arrays.fill(dp, amount + 1);
        dp[0] = 0;
        for (int i = 1; i <= amount; i++)
            for (int j = 0; j < coins.length; j++)
                if (coins[j] <= i)
                    dp[i] = Math.min(dp[i], dp[i-coins[j]] + 1);
        return dp[amount] == amount + 1 ? -1 : dp[amount];
    }
```

### 139.单词拆分

```java
    public boolean wordBreak(String s, List<String> wordDict) {
        Set<String> set = new HashSet<>(wordDict);
        int n = s.length();
        boolean[] dp = new boolean[n+1];
        dp[0] = true;
        for (int i = 1; i <= n; i++)
            for (int j = 0; j < i; j++)
                if (dp[j] && set.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
        return dp[n];
    }
```

### 300.最长递增子序列

```java
    public int lengthOfLIS(int[] nums) {
        int[] dp = new int[nums.length];
        Arrays.fill(dp, 1);
        int res = 1;

        for (int i = 1; i < nums.length; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] < nums[i]) {
                    dp[i] = Math.max(dp[i], dp[j] + 1);
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }
```

### 152.乘积最大子数组

```java
    public int maxProduct(int[] nums) {
        int maxDP = nums[0], minDP = nums[0], res = nums[0];
        for (int i = 1; i < nums.length; i++) {
            int curMaxDP = maxDP, curMinDP = minDP;
            maxDP = Math.max(nums[i], Math.max(nums[i] * curMaxDP, nums[i] * curMinDP));
            minDP = Math.max(nums[i], Math.min(nums[i] * curMaxDP, nums[i] * curMinDP));
            res = Math.max(res, maxDP);
        }
        return res;
    }
```

### 416.分割等和子集

```java
    public boolean canPartition(int[] nums) {
        int sum = 0;
        for (int num : nums) sum += num;
        if (sum % 2 == 1) return false;
        int target =  sum / 2;
        boolean[] dp = new boolean[target+1];
        dp[0] = true;
        for (int num : nums)
            for (int j = target; j >= num; j--)
                dp[j] = dp[j] || dp[j - num];
        return dp[target];
    }
```

### 32.最长有效括号

```java
    public int longestValidParentheses(String s) {
        int length = s.length();
        if (s == null || length < 2) return 0;
        int[] dp = new int[length];
        int res = 0;
        for (int i = 1; i < length; i++) {
            if (s.charAt(i) == ')') {
                if (s.charAt(i-1) == '(') {
                    dp[i] = (i >= 2 ? dp[i-2] : 0) + 2;
                } else if (dp[i-1] > 0) {
                    int j = i - dp[i-1] - 1;
                    if (j >= 0 && s.charAt(j) == '(') {
                        dp[i] = dp[i-1] + 2;
                        if (j > 0) {
                            dp[i] += dp[j-1];
                        }
                    }
                }
            }
            res = Math.max(res, dp[i]);
        }
        return res;
    }
```

```java
	public int longestValidParentheses(String s) {
       if(s == null || s.length() < 2) return 0;
       int len = s.length(), res = 0;
       Stack<Integer> stack = new Stack<>();
       stack.push(-1);
       for(int i = 0; i < len; i++) {
           if(s.charAt(i) == '(') {
               stack.push(i);
           }else {
               stack.pop();
               if(stack.empty()) {
                   stack.push(i);
               }else {
                   res = Math.max(res, i - stack.peek());
               }
           }
       }
       return res;
   }
```

## 多维动态规划

### 62.不同路径

```java
    public int uniquePaths(int m, int n) {
        int[][] dp = new int[m][n];
        for (int i = 0; i < m; i++) dp[i][0] = 1;
        for (int i = 0; i < n; i++) dp[0][i] = 1;
        for (int i = 1; i < m; i++)
            for (int j = 1; j < n; j++)
                dp[i][j] = dp[i-1][j] + dp[i][j-1];
        return dp[m-1][n-1];
    }
```

### 64.最小路径和

```java
    public int minPathSum(int[][] grid) {
        int[][] dp = new int[grid.length][grid[0].length];
        dp[0][0] = grid[0][0];
        for (int i = 1; i < grid.length; i++)
        	dp[i][0] = dp[i-1][0] + grid[i][0];
        for (int i = 1; i < grid[0].length; i++)
        	dp[0][i] = dp[0][i-1] + grid[0][i];
        for (int i = 1; i < grid.length; i++)
            for (int j = 1; j < grid[0].length; j++)
                dp[i][j] = grid[i][j] + Math.min(dp[i-1][j], dp[i][j-1]);
        return dp[grid.length-1][grid[0].length-1];
    }
```

### 5.最长回文子串

```java
	// 动态规划法
    public String longestPalindrome(String s) {
        if (s.length() < 2) return s;
        int len = s.length();
        int maxLen = 1, startIdx = 0, endIdx = 0;
        boolean[][] dp = new boolean[len][len];

        for (int r = 1; r < len; r++) {
            for (int l = 0; l < r; l++) {
                if (s.charAt(l) == s.charAt(r) && (r - l <= 2 || dp[l+1][r-1])){
                    dp[l][r] = true;
                    if (r - l + 1 > maxLen) {
                        startIdx = l;
                        endIdx = r;
                        maxLen = r - l + 1;
                    }
                }
            }
        }
        return s.substring(startIdx, endIdx+1);
    }

    // 中心拓展法
    public String longestPalindrome(String s) {
        int start = 0, end = 0, length = s.length();
        for (int i = 0; i < length; i++) {
            int lenOdd = solve(s, i, i, length);
            int lenEven = solve(s, i, i + 1, length);
            int maxLen = Math.max(lenOdd, lenEven);
            if (maxLen > end - start) {
                start = i - (maxLen - 1) / 2;
                end = i + maxLen / 2;
            }
        }
        return s.substring(start, end + 1);
    }
    public int solve(String s, int left, int right, int length) {
        while (left >= 0 && right < length && s.charAt(left) == s.charAt(right)) {
            left--;
            right++;
        }
        return right - left - 1;
    }
```

### 1143.最长公共子序列

```java
    public int longestCommonSubsequence(String text1, String text2) {
        int m = text1.length(), n = text2.length();
        int[][] dp = new int[m+1][n+1];
        for (int i = 1; i <= m; i++)
            for (int j = 1; j <= n; j++)
                if (text1.charAt(i-1) == text2.charAt(j-1))
                    dp[i][j] = dp[i-1][j-1] + 1;
                else
                    dp[i][j] = Math.max(dp[i-1][j], dp[i][j-1]);
        return dp[m][n];
    }
```

### 72.编辑距离

```java
    public int minDistance(String word1, String word2) {
        int m = word1.length(), n = word2.length();
        int[][] dp = new int[m+1][n+1];
        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int i = 0; i <= n; i++) dp[0][i] = i;
        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++)
                if (word1.charAt(i-1) == word2.charAt(j-1))
                    dp[i][j] = dp[i-1][j-1];
                else
                    dp[i][j] = Math.min(dp[i-1][j-1], Math.min(dp[i-1][j], dp[i][j-1])) + 1;
        }
        return dp[m][n];
    }
```

## 技巧

### 136.只出现一次的数字

```java
    public int singleNumber(int[] nums) {
        int res = 0;
        for (int num : nums)
            res ^= num;
        return res;
    }
```

### 169.多数元素

```java
    public int majorityElement(int[] nums) {
        Arrays.sort(nums);
        return nums[nums.length / 2];
    }
```

### 75.颜色分类

```java
    public void sortColors(int[] nums) {
        int left = 0, right = nums.length - 1, i = 0;
        while (i <= right) {
            if (nums[i] == 0){
                swap(nums, i, left);
                left++;
                i++;
            } else if (nums[i] == 1) {
                i++;
            } else {
                swap(nums, i, right);
                right--;
            }
        }
    }
    public void swap(int[] nums, int i, int j) {
        int T = nums[i];
        nums[i] = nums[j];
        nums[j] = T;
    }
```

### 31.下一个排列

```java
    public void nextPermutation(int[] nums) {
        int i = nums.length - 2;
        while (i >= 0 && nums[i] >= nums[i+1])
        	i--;
        if (i >= 0) {
            int j = nums.length - 1;
            while (nums[j] <= nums[i])
            	j--;
            swap(nums, i, j);
        }
        reverse(nums, i+1, nums.length - 1);
    }
    public void swap(int[] nums, int i, int j) {
        int T = nums[i];
        nums[i] = nums[j];
        nums[j] = T;
    }
    public void reverse(int[] nums, int left, int right) {
        while (left < right)
        	swap(nums, left++, right--);
    }
```

### 287.寻找重复数

```java
    public int findDuplicate(int[] nums) {
        int slow = 0, fast = 0;
        while (true) {
            fast = nums[fast];
            fast = nums[fast];
            slow = nums[slow];
            if (fast == slow) break;
        }
        fast = 0;
        while (fast != slow) {
            fast = nums[fast];
            slow = nums[slow];
        }
        return fast;
    }
```
