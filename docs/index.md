# A deep learning-based compatibility score for reconstruction of strip-shredded text documents
### [Thiago M. Paixão¹²](http://sites.google.com/site/professorpx), Rodrigo F. Berriel², Maria C. S. Boeres², Claudine Badue², Alberto F. De Souza² and Thiago Oliveira-Santos²
#### ¹Instituto Federal do Espírito Santo, ²Universidade Federal do Espírito Santo
#### paixao at gmail dot com

------
We also used the trained model to reconstruct the full collection of mixed documents (aprox. 1850 strips).
The resulting reconstruction is available [here](https://daringfireball.net/projects/markdown/).
---

#### Main dependencies:
* [docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
* [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).
```
No language indicated, so no syntax highlighting. 
But let's throw in a <b>tag</b>.
```
With [GitHub Pages](https://pages.github.com), you just write things in
[Markdown](https://daringfireball.net/projects/markdown/),
[GitHub](https://github.com) hosts the site for you, and you just push
material to your GitHub repository with `git add`, `git commit`, and
`git push`.
The sites use [Jekyll](https://jekyllrb.com/), a
[ruby](https://www.ruby-lang.org/en/) [gem](https://rubygems.org/), to
convert Markdown files to html, and this part is done
automatically when you push the materials to the `gh-pages` branch
of a GitHub repository.


The [GitHub](https://pages.github.com) and
[Jekyll](https://jekyllrb.com) documentation is great, but I thought it
would be useful to have a minimal tutorial, for those who just want to
get going immediately with a simple site. To some readers, what GitHub
has might be simpler and more direct.  But if you just want to create
a site like the one you're looking at now, read on.

Start by reading the [Overview page](pages/overview.html), which
explains the basic structure of these sites. Then read
[how to make an independent website](pages/independent_site.html). Then
read any of the other things, such as
[how to test your site locally](pages/local_test.html).

- [Overview](pages/overview.html)
- [Making an independent website](pages/independent_site.html)
- [Making a personal site](pages/user_site.html)
- [Making a site for a project](pages/project_site.html)
- [Making a jekyll-free site](pages/nojekyll.html)
- [Testing your site locally](pages/local_test.html)
- [Resources](pages/resources.html)

If anything here is confusing (or _wrong_!), or if I've missed
important details, please
[submit an issue](https://github.com/kbroman/simple_site/issues), or (even
better) fork [the GitHub repository for this website](https://github.com/kbroman/simple_site),
make modifications, and submit a pull request.

---

The source for this minimal tutorial is [on github](https://github.com/kbroman/simple_site).

Also see my [tutorials](https://kbroman.org/pages/tutorials) on
[git/github](https://kbroman.org/github_tutorial),
[GNU make](https://kbroman.org/minimal_make),
[knitr](https://kbroman.org/knitr_knutshell),
[R packages](https://kbroman.org/pkg_primer),
[data organization](https://kbroman.org/dataorg),
and [reproducible research](https://kbroman.org/steps2rr).
