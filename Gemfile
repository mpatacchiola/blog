source "https://rubygems.org"
ruby RUBY_VERSION

# Hello! This is where you manage which Jekyll version is used to run.
# When you want to use a different version, change it below, save the
# file and run `bundle install`. Run Jekyll with `bundle exec`, like so:
#
#     bundle exec jekyll serve
#
# This will help ensure the proper Jekyll version is running.
# Happy Jekylling!
#gem "jekyll", "3.3.0"
# Patch for GitHub alert
gem "jekyll", ">= 3.6.3"


# This is the default theme for new Jekyll sites. You may change this to anything you like.
gem "minima", "~> 2.0"

# If you want to use GitHub Pages, remove the "gem "jekyll"" above and
# uncomment the line below. To upgrade, run `bundle update github-pages`.
# gem "github-pages", group: :jekyll_plugins

# If you have any plugins, put them here!
group :jekyll_plugins do
   gem "jekyll-feed", "~> 0.6"
   gem 'jekyll-seo-tag'
end

# Patch for GitHub alert
# The kramdown gem before 2.3.0 for Ruby processes the template option inside Kramdown 
#documents by default, which allows unintended read access (such as template="/etc/passwd") 
#or unintended embedded Ruby code execution.
gem "kramdown", ">= 2.3.0"
