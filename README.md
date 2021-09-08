# Welcome
## If you're reading this from Github or your Markdown viewer:
Please follow the following steps to install the ML@B Research Vault. Then, read the section below the instructions in Obsidian.
1. Clone [this repository](https://github.com/mlberkeley/reading-group) to a local folder
2. Download [Obsidian](https://obsidian.ml/)
3. Open Obsidian and click the “Open” button next to “Open folder as vault” ![Open Instructions | 200](Images/README/open.png)
4. Navigate to the folder you cloned the `reading-group` repository into ![Navigation Instructions | 400](Images/README/navigate.png)
5. You should see the following Directed Acyclic Graph (DAG) in the Obsidian Graph Viewer ![DAG | 300](Images/README/dag.png)
6. Try filtering for parts of the DAG you're interested in
	1. Open the "Filters" drop down list to the top left of the graph viewer and click on the search bar 
	2. Try filtering for the files in the base folder "Linear Algebra" using the "path:" option 
	![[Images/README/filterfile.png | 400]]
	4. Try adding a tag (as \#\<tag name\>, e.g. #read) to a few articles and filter down to these articles using the "tag:" option 
	![[Images/README/readtags.png | 400]] ![[Images/README/filtertag.png | 400]]
	4. Try filtering for articles either in "Linear Algebra" OR tagged with #read by concatenating the relevant "file:" and "tag:" searches with the "OR" boolean operator. Equivalently, use the "AND" boolean operator to filter articles tagged #read within "Linear Algebra". ![[Images/README/filteror.png | 400]]
7. Try grouping the DAG
	1. Open the "Groups" drop down list to the top left of the graph viewer, press “new group,” and explore the possible criteria 
	![[Images/README/groups.png | 200]]
	3. Try creating a group for each base folder in the `reading-group` repo using the "path:" option 
	![[Images/README/groupfile.png | 400]]
	4. Try adding a tag (as \#\<tag name\>, e.g. #read) to a few articles and visualizing this group of articles with the "tag:" option 
	![[Images/README/readtags.png | 400]] ![[Images/README/grouptag.png | 400]]
	6. Try creating a group of articles either in "Linear Algebra" OR tagged with #read by concatenating the relevant "file:" and "tag:" searches with the "OR" boolean operator. Equivalently, use the "AND" boolean operator to group articles tagged #read within "Linear Algebra" ![[Images/README/groupor.png | 400]]
8. Click the preview button in the top right corner with the Cmd (mac) or Ctrl (windows) button held (or whatever the obsidian popup says) to open a preview in a separate pane. Try adding the following the following and watch the preview update.
	1. A numbered list: "1. blah \n 2. blah" ->
		1. blah
		2. blah
	2. A bulleted list: "- blah \n - blah" ->
		- blah
		- blah
	3. In-text latex: "\$a = 1\$" -> $a = 1$
	4. Latex on its own line: "\$\$a = 1\$\$" -> $$a = 1$$
	5. An image, like the following one to help you find the preview pane: "!\[{Name} | {width}\]({path})" -> 
	"!\[Preview | 200\](Images/README/preview.png)" -> ![Preview | 200](Images/README/preview.png)
9. If you create something you'd like to be added to the master repository, create a pull request [on Github!](https://www.github.com/mlberkeley/reading-group/pulls)

## If you're reading this in Obsidian:
You're in the right place. Welcome to the ML@B Research vault! This is a folder of markdown files (see [Markdown Guide](https://www.markdownguide.org)) meant to be a central repository for ML@B research knowledge. It will be organized as an [Obsidian](https://obsidian.md/) vault, so be sure to download Obsidian if you wish to use it. This folder will be shared with all ML@B members and updated via `git`. It is intended to serve as
1. A structured, piece-by-piece educational resource for ML@B members unfamiliar with many areas of ML
2. An unstructured venue for tinkering with new ideas
3. A tool for transforming unstructured exploration into structured, absorbable knowledge

As more is decided regarding the structure of our notes, we will add to this note explaining the organization of the vault, and giving instructions to new members about how to get involved.

## Notation
- $\mathcal{D}(v)$ refers to a distribution over the variable $v$