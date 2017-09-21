def plot_linear_regression(x, y, slope, intercept, fig_title='',
                           x_title='', y_title='', show_fig=False):
    import plotly.offline as po
    import plotly.graph_objs as go

    points_trace = go.Scatter(
        x=x,
        y=y,
        mode='markers',
        name='Documents'
    )

    y_line = intercept + slope * x

    line_trace = go.Scatter(
        x=x,
        y=y_line,
        name='Fitted line',
        line=dict(
            color=('red'),
            width=4,
            dash='dot'
        )
    )

    fig = go.Figure(data=go.Data([points_trace, line_trace]),
                    layout=go.Layout(
                        xaxis=dict(
                            title=x_title,
                            autorange=True,
                            showgrid=False,
                            zeroline=True,
                            showline=False,
                            autotick=True,
                            # ticks='',
                            # showticklabels=False
                        ),
                        yaxis=dict(
                            title=y_title,
                            autorange=True,
                            showgrid=False,
                            zeroline=True,
                            showline=False,
                            autotick=True,
                            # ticks='',
                            # showticklabels=False
                        ),
                        title=fig_title,
                        titlefont=dict(size=16),
                        # showlegend=False,
                        # width=900,
                        # height=650,
                        paper_bgcolor='rgba(238,238,238,1)',
                        plot_bgcolor='rgba(238,238,238,1)',
                        hovermode='closest',
                        # margin=dict(b=20, l=5, r=5, t=40)
                    ))
    if not show_fig:
        output = 'div'
    else:
        output = 'file'
    return po.plot(fig, filename='line.html',
                   auto_open=show_fig,
                   output_type=output)


def plot_taxonomy_coverage(the, fig_title='', show_fig=False, bg_color='white',
                           div_width=None, div_height=None):
    import plotly.offline as po
    import plotly.graph_objs as go

    G, pos = the.plot_layout()
    cpt_uris = the.get_all_concepts()
    covered = dict()
    children_covered = dict()
    for cpt_uri in cpt_uris:
        cpt_own_freq = int(the.get_own_freq(cpt_uri, def_value=0))
        cpt_cum_frequency = int(the.get_cumulative_freq(cpt_uri, def_value=0))
        cpt_pl = str(the.get_pref_label(cpt_uri))
        if cpt_own_freq:
            covered[cpt_pl] = cpt_own_freq
        if cpt_cum_frequency:
            children_covered[cpt_pl] = cpt_cum_frequency - cpt_own_freq


    Xe = []
    Ye = []
    for edge in G.edges():
        Xe += [pos[edge[0]][0], pos[edge[1]][0], None]
        Ye += [pos[edge[0]][1], pos[edge[1]][1], None]

    lines = go.Scatter(x=Xe,
                       y=Ye,
                       mode='lines',
                       line=dict(color='rgb(210,210,210)', width=2),
                       marker=dict(symbol='triangle'),
                       hoverinfo='none')
    dots = go.Scatter(x=[],
                      y=[],
                      mode='markers',
                      name='',
                      marker=dict(
                          symbol='dot',
                          size=[],
                          color=[],
                          line=dict(color='rgb(50,50,50)', width=1)
                      ),
                      text=[],
                      hoverinfo='text',
                      # opacity=0.8
                      )
    max_freq = max(covered.values())
    for node in G.nodes():
        x, y = pos[node]
        node_info = '{}, own frequency: {}, children frequency: {}'.format(
            str(node),
            covered[node] if node in covered else 0,
            children_covered[node] if node in children_covered else 0
        )
        scaled_size = (45*covered[node]/max_freq + 5) if node in covered else 15
        dots['x'].append(x)
        dots['y'].append(y)
        dots['marker']['color'].append(
            'blue' if node in covered else
                ('orange' if node in children_covered else 'grey')
        )
        dots['marker']['size'].append(scaled_size)
        dots['text'].append(node_info)

    axis = dict(
        showline=False, zeroline=False, showgrid=False, showticklabels=False
    )

    layout = go.Layout(
        title=fig_title,
        font=dict(size=12),
        showlegend=False,
        width=div_width,
        height=div_height,
        xaxis=go.XAxis(axis),
        yaxis=go.YAxis(axis),
        margin=dict(l=40, r=40, b=85, t=100),
        hovermode='closest',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
    )
    data = go.Data([lines, dots])
    fig = dict(data=data, layout=layout)
    if not show_fig:
        output = 'div'
    else:
        output = 'file'
    return po.plot(fig, filename='taxonomy_coverage.html',
                   auto_open=show_fig,
                   output_type=output)